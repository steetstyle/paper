"""Finish runs whose owning process is gone.

An ingestion run is marked ``running`` and then, if the process that owned it
dies — a closed laptop, a killed shell, a lost connection mid-download — nothing
ever moves it on. The row keeps saying ``running`` forever, and a list of runs
becomes a list of lies: 38 of them were, the oldest for over a day, with no way to
tell them apart from work genuinely in progress.

Liveness is inferred from the step records rather than from a heartbeat column.
Every completed step is written to ``pipeline_step_runs``, so a run whose newest
step finished long ago has demonstrably stopped making progress, whatever happened
to it. A run with *no* step record at all falls back to its creation time.

The bound is deliberately generous (:attr:`IngestionSettings.stale_run_hours`,
six hours by default) because the cost is asymmetric. A false positive costs a
re-ingest, and the re-ingest reuses the stored blob and the stored extracted
markdown, so it is minutes rather than a full run. A false *negative* — leaving a
dead run looking busy — is what produced the 38 in the first place.

Reaping is explicit, never a side effect of listing. Reading state should not
mutate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IngestionRun, PipelineStepRun
from app.domain.enums import RunStatus
from app.logging import get_logger

logger = get_logger(__name__)

#: Statuses that mean "still going". Everything else is already terminal and
#: needs no reaping.
_UNFINISHED = (RunStatus.PENDING.value, RunStatus.RUNNING.value)


@dataclass(frozen=True, slots=True)
class StaleRun:
    """One run presumed dead, with the evidence for that."""

    id: str
    arxiv_id: str | None
    doc_key: str | None
    status: str
    created_at: datetime | None
    last_progress_at: datetime | None
    steps_completed: int

    @property
    def age_hours(self) -> float | None:
        """How long since anything touched it. ``None`` if it never was."""
        moment = self.last_progress_at or self.created_at
        if moment is None:
            return None
        return round((_utcnow() - moment).total_seconds() / 3600, 1)

    @property
    def target(self) -> str:
        """What to re-run: the arXiv id, else the document's handle."""
        return self.arxiv_id or self.doc_key or self.id

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.id,
            "target": self.target,
            "arxiv_id": self.arxiv_id,
            "doc_key": self.doc_key,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_progress_at": (
                self.last_progress_at.isoformat() if self.last_progress_at else None
            ),
            "age_hours": self.age_hours,
            "steps_completed": self.steps_completed,
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


def resume_instead_of_skip(previous_status: str | None) -> bool:
    """Whether a re-run of this target should resume rather than be skipped.

    A document that is already in the corpus is normally refused as a duplicate.
    That is wrong exactly when the last attempt at it did not finish: the corpus
    has a row for it, the row is incomplete, and refusing the retry leaves no way
    forward except ``--force`` — which discards the stored markdown and pays for
    MinerU again. Measured: a 140-page book takes ~96s to extract and ~18ms to
    reload, so the choice is between seconds and minutes on the one operation
    nobody wants to repeat by hand.

    ``PARTIAL`` counts as unfinished: it is a run that lost a step, which is
    precisely the state worth retrying.
    """
    if previous_status is None:
        return True
    return RunStatus(previous_status) not in {
        RunStatus.SUCCEEDED,
        RunStatus.SKIPPED,
    }


async def find_stale_runs(
    session: AsyncSession, *, older_than_hours: float = 6.0
) -> list[StaleRun]:
    """Unfinished runs that nothing has touched for ``older_than_hours``.

    Ordered oldest-first, so reaping the oldest first also means the report leads
    with whatever has been stuck longest.
    """
    cutoff = _utcnow() - timedelta(hours=older_than_hours)
    rows = (
        await session.execute(
            select(IngestionRun).where(IngestionRun.status.in_(_UNFINISHED))
        )
    ).scalars().all()

    stale: list[StaleRun] = []
    for run in rows:
        last_step = (
            await session.execute(
                select(func.max(PipelineStepRun.finished_at)).where(
                    PipelineStepRun.run_id == run.id
                )
            )
        ).scalar()
        last_progress = _aware(last_step) or _aware(run.created_at)
        if last_progress is None or last_progress > cutoff:
            continue
        completed = (
            await session.execute(
                select(func.count())
                .select_from(PipelineStepRun)
                .where(PipelineStepRun.run_id == run.id)
            )
        ).scalar_one()
        stale.append(
            StaleRun(
                id=run.id,
                arxiv_id=run.arxiv_id,
                doc_key=run.doc_key,
                status=run.status,
                created_at=_aware(run.created_at),
                last_progress_at=_aware(last_step),
                steps_completed=int(completed or 0),
            )
        )
    stale.sort(key=lambda item: item.last_progress_at or _utcnow())
    return stale


async def reap_stale_runs(
    session: AsyncSession, *, older_than_hours: float = 6.0, dry_run: bool = True
) -> dict[str, object]:
    """Mark abandoned runs terminal. Returns what was found and what changed.

    ``dry_run`` defaults to ``True`` because this decides the fate of rows that
    record what actually happened; a report first and a decision second is the
    right default for something that ends a run's history.
    """
    stale = await find_stale_runs(session, older_than_hours=older_than_hours)
    if not stale:
        return {"found": 0, "reaped": 0, "targets": [], "runs": [], "dry_run": dry_run}

    if dry_run:
        return {
            "found": len(stale),
            "reaped": 0,
            "targets": _by_target(stale),
            "runs": [item.as_dict() for item in stale],
            "dry_run": True,
        }

    for item in stale:
        run = await session.get(IngestionRun, item.id)
        if run is None:
            continue
        # Close any step that was mid-flight: the process holding it is gone, and
        # a row with no `finished_at` is what makes a run look like it is still
        # working. Without this, `list_ingest_runs` keeps reporting it as active.
        await session.execute(
            update(PipelineStepRun)
            .where(PipelineStepRun.run_id == item.id, PipelineStepRun.finished_at.is_(None))
            .values(finished_at=_utcnow())
        )
        run.status = RunStatus.ABANDONED.value
        run.finished_at = _utcnow()
        run.error = (
            f"abandoned: nothing has touched this run for {item.age_hours}h; "
            f"its process is gone. Re-run `paper ingest {item.target}` to resume "
            "from the stored artefacts"
        )
    await session.commit()
    logger.info("stale_runs_reaped", extra={"count": len(stale)})
    return {
        "found": len(stale),
        "reaped": len(stale),
        "targets": _by_target(stale),
        "runs": [item.as_dict() for item in stale],
        "dry_run": False,
    }


def _by_target(stale: list[StaleRun]) -> list[dict[str, object]]:
    """Group by what to re-run, newest progress last.

    The grouping is the actionable part: 40 stuck rows can be 3 papers. Measured on
    the live corpus — two abandoned runs for ``2302.07175``, because a retry
    creates a new run and leaves the old one where it was. "Re-run this" is a
    question about targets, not rows.
    """
    grouped: dict[str, list[StaleRun]] = {}
    for item in stale:
        grouped.setdefault(item.target, []).append(item)
    # Sorted before it becomes a dict: the ages are floats here and `object`
    # afterwards, and the ordering is the whole point of the grouping.
    rows = sorted(
        grouped.items(),
        key=lambda pair: (-max(i.age_hours or 0.0 for i in pair[1]), pair[0]),
    )
    return [
        {
            "target": target,
            "stuck_runs": len(items),
            "arxiv_id": items[0].arxiv_id,
            "doc_key": items[0].doc_key,
            "age_hours": max(i.age_hours or 0.0 for i in items),
            "steps_completed": max(i.steps_completed for i in items),
            "run_ids": [i.id for i in items],
        }
        for target, items in rows
    ]


def _aware(value: datetime | None) -> datetime | None:
    """Coerce to an aware UTC datetime, or ``None``.

    SQLite hands back naive timestamps and Postgres hands back aware ones, and
    comparing the two raises. Coercing here is what lets one implementation of the
    staleness rule work on both.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
