"""Pipeline runner.

Owns step orchestration, timing, per-step persistence and failure policy:

* a ``fatal`` step failure aborts the run
* a non-fatal failure is recorded and the pipeline continues
* every step outcome lands in ``pipeline_step_runs`` for observability
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from typing import Any

from paper_app.domain.enums import RunStatus, StepStatus
from paper_app.logging import bind_run_id, get_logger
from paper_app.pipeline.context import PipelineContext
from paper_app.pipeline.steps import Step, StepResult

logger = get_logger(__name__)


class PipelineRunner:
    def __init__(self, steps: Sequence[Step], *, continue_on_error: bool = True) -> None:
        self._steps = list(steps)
        self._continue_on_error = continue_on_error

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        bind_run_id(ctx.run_id)
        await ctx.runs.mark_running(ctx.run_id, ctx.paper_id)

        attempts: dict[str, int] = {}
        fatal: Exception | None = None

        for step in self._steps:
            attempts[step.name] = attempts.get(step.name, 0) + 1
            started = time.perf_counter()
            status = StepStatus.SUCCEEDED
            error: str | None = None
            payload: dict[str, Any] = {}
            step_failed = False

            try:
                skip_reason = None if ctx.force else step.should_skip(ctx)
                if skip_reason:
                    result: StepResult = StepResult.skip(skip_reason)
                    status = StepStatus.SKIPPED
                else:
                    result = await step.run(ctx)
                payload = dict(result.data or {})

                if result.skipped:
                    status = StepStatus.SKIPPED
                elif not result.ok:
                    status = StepStatus.FAILED
                    step_failed = True
                    error = str(payload.get("error", "step failed"))
                    if result.fatal or not self._continue_on_error:
                        fatal = RuntimeError(f"{step.name}: {error}")
                        logger.error("pipeline_step_fatal", extra={"step": step.name, "error": error})
                    else:
                        logger.warning("pipeline_step_degraded", extra={"step": step.name, "error": error})

            except Exception as exc:  # noqa: BLE001 - recorded then re-raised
                status = StepStatus.FAILED
                step_failed = True
                error = f"{type(exc).__name__}: {exc}"
                fatal = exc
                logger.error(
                    "pipeline_step_raised",
                    extra={"step": step.name, "error": error, "traceback": traceback.format_exc()},
                )

            duration_ms = (time.perf_counter() - started) * 1000
            ctx.timings[step.name] = round(duration_ms, 2)
            await ctx.runs.add_step(
                ctx.run_id,
                name=step.name,
                status=status.value,
                attempt=attempts[step.name],
                duration_ms=round(duration_ms, 2),
                error=error,
                meta=payload,
            )
            await ctx.session.commit()

            if fatal is not None:
                break
            if step_failed:
                continue

        final_status = self._final_status(ctx, fatal)
        await ctx.runs.finish(
            ctx.run_id,
            final_status,
            error=str(fatal) if fatal is not None else None,
            timings=ctx.timings,
            chunk_count=len(ctx.chunks),
            embedding_count=ctx.embedded_count,
            content_kind=ctx.content_kind.value if ctx.content_kind else None,
        )
        await ctx.session.commit()
        logger.info(
            "pipeline_finished",
            extra={
                "run_id": ctx.run_id,
                "status": final_status.value,
                "chunks": len(ctx.chunks),
                "embeddings": ctx.embedded_count,
                "warnings": ctx.warnings,
            },
        )
        return ctx

    @staticmethod
    def _final_status(ctx: PipelineContext, fatal: Exception | None) -> RunStatus:
        if fatal is not None and not ctx.chunks:
            return RunStatus.FAILED
        if fatal is not None or ctx.warnings:
            return RunStatus.PARTIAL
        return RunStatus.SUCCEEDED