"""Read-only mode: the CLI's promise that looking cannot change anything.

The mode exists for an agent or a shell that should be able to inspect the corpus
without being trusted to leave it alone. Its whole value is that the promise is
mechanical rather than a matter of which commands the caller remembered to avoid,
so the tests here are about *coverage*: every command Typer exposes must be
accounted for, and an unaccounted one fails a test instead of quietly writing.

That is the failure this guards against. A mode that refuses most writes and
misses one is worse than no mode, because the caller stops checking.
"""

from __future__ import annotations

import pytest
import typer
from typer.main import get_command

from app.cli import READ_ONLY_COMMANDS, app


def _all_commands(group, prefix: str = "") -> dict[str, object]:  # noqa: ANN001
    """Every command Typer exposes, keyed by its path (``db:upgrade``)."""
    found: dict[str, object] = {}
    for name, command in getattr(group, "commands", {}).items():
        path = f"{prefix}{name}"
        if hasattr(command, "commands"):
            found.update(_all_commands(command, f"{path}:"))
        else:
            found[path] = command
    return found


ALL_COMMANDS = _all_commands(get_command(app))


def _writes_flags(command) -> tuple[str, ...]:  # noqa: ANN001
    """The flags that make a command write. Empty means "always"."""
    return getattr(command.callback, "_writes_flags", None) or ()


def _is_writer(command) -> bool:  # noqa: ANN001
    """Unconditionally writes."""
    return (
        getattr(command.callback, "_writes_command", None) is not None
        and not _writes_flags(command)
    )


def _is_conditional_writer(command) -> bool:  # noqa: ANN001
    """Writes only when one of its flags is set — guarded on that flag."""
    return bool(getattr(command.callback, "_writes_command", None)) and bool(
        _writes_flags(command)
    )


class TestCompleteness:
    def test_every_command_is_accounted_for(self) -> None:
        """The invariant. A new command that is neither decorated nor allowlisted
        fails here rather than shipping an unguarded write."""
        unaccounted = [
            path
            for path, command in ALL_COMMANDS.items()
            if not _is_writer(command)
            and not _is_conditional_writer(command)
            and path not in READ_ONLY_COMMANDS
        ]
        assert unaccounted == [], (
            f"commands neither marked @writes nor listed as read-only: {unaccounted}"
        )

    def test_the_allowlist_holds_no_writers(self) -> None:
        """A command cannot be both, or the guard is decorative for it."""
        both = [
            path for path in READ_ONLY_COMMANDS if _is_writer(ALL_COMMANDS.get(path))
        ]
        assert both == []

    def test_a_conditional_writer_must_actually_be_guarded(self) -> None:
        """Being in the read allowlist is only safe for a conditional writer
        because its flag is guarded. A command marked conditionally but not
        guarded would delete rows under ``--read-only``."""
        for path, command in ALL_COMMANDS.items():
            if _is_conditional_writer(command):
                assert getattr(command.callback, "_read_only_guarded", False), path

    def test_the_allowlist_names_nothing_that_does_not_exist(self) -> None:
        """A stale entry is a hole in the test above: it would let a *renamed*
        command through unaccounted while still looking complete."""
        missing = sorted(set(READ_ONLY_COMMANDS) - set(ALL_COMMANDS))
        assert missing == []

    def test_the_survey_is_not_trivially_empty(self) -> None:
        """Guards the survey itself. If `get_command` changed shape and returned
        nothing, every assertion above would pass vacuously."""
        assert len(ALL_COMMANDS) > 15
        assert sum(1 for c in ALL_COMMANDS.values() if _is_writer(c)) >= 10


class TestRefusal:
    @pytest.mark.parametrize(
        "path",
        [
            "ingest",
            "harvest",
            "reembed",
            "kinds",
            "projects:new",
            "projects:add",
            "projects:rm",
            "projects:read",
            "spaces:add",
            "spaces:activate",
            "spaces:rm",
            "db:upgrade",
            "db:downgrade",
        ],
    )
    def test_a_writer_refuses_before_its_body_runs(self, path: str) -> None:
        """Called with no arguments on purpose: the guard fires before the
        signature is ever satisfied, which is exactly the claim — refused first,
        work never started."""
        from app.cli import read_mode  # noqa: PLC0415

        command = ALL_COMMANDS[path]
        read_mode(True)
        try:
            with pytest.raises(typer.Exit) as caught:
                command.callback()
        finally:
            read_mode(False)
        assert caught.value.exit_code == 3

    @pytest.mark.parametrize("path", sorted(READ_ONLY_COMMANDS))
    def test_a_reader_is_never_refused_by_the_guard(self, path: str) -> None:
        """A read command must not inherit the guard by accident; the test asserts
        the marker, since invoking them all would need a database."""
        assert not _is_writer(ALL_COMMANDS[path])


class TestConditionalWriters:
    """Commands that read by default and write only when asked twice.

    ``paper runs --reap`` reports; ``--apply`` closes. Marking the command a writer
    would refuse the report, which is the safe half; leaving it a reader would let
    ``--read-only paper runs --reap --apply`` delete rows, which breaks the only
    promise the mode makes. So the guard is on the flag.
    """

    @pytest.mark.parametrize(
        ("path", "flag"),
        [("runs", "apply"), ("sections", "apply")],
    )
    def test_the_flag_is_what_the_guard_reads(self, path: str, flag: str) -> None:
        command = ALL_COMMANDS[path]
        assert getattr(command.callback, "_writes_flags", None) == (flag,)

    def test_the_dry_run_is_allowed_and_the_apply_is_not(self) -> None:
        """Exercised on a stand-in rather than on the real commands, which would
        open a database. What is under test is the guard, not the report."""
        from app.cli import read_mode, writes_when  # noqa: PLC0415

        ran: list[bool] = []

        @writes_when("demo", "apply")
        def command(reap: bool = False, apply: bool = False) -> None:  # noqa: A002
            ran.append(apply)

        read_mode(True)
        try:
            command(reap=True)  # the report
            with pytest.raises(typer.Exit) as caught:
                command(reap=True, apply=True)
        finally:
            read_mode(False)
        assert ran == [False], "the body must not run when refused"
        assert caught.value.exit_code == 3

    def test_outside_read_mode_the_flag_writes_normally(self) -> None:
        from app.cli import writes_when  # noqa: PLC0415

        ran: list[bool] = []

        @writes_when("demo", "apply")
        def command(apply: bool = False) -> None:  # noqa: A002
            ran.append(apply)

        command(apply=True)
        assert ran == [True]

    def test_both_are_still_in_the_read_allowlist(self) -> None:
        """They read by default, so they belong there; the flag guard is what makes
        that safe."""
        assert {"runs", "sections"} <= READ_ONLY_COMMANDS

    def test_the_global_flag_is_a_real_option_on_the_root_command(self) -> None:
        """Registered on the root group, so it shows in `paper --help` and
        composes with any command. Asserted on the command tree rather than by
        invoking it: driving the whole app through CliRunner imports every command
        module, which turns this into a test of Click's plumbing."""
        opts = {opt for param in get_command(app).params for opt in param.opts}
        assert "--read-only" in opts

    def test_the_flag_sets_the_mode_and_clears_again(self) -> None:
        from app.cli import _global_options, is_read_only, read_mode  # noqa: PLC0415

        assert is_read_only() is False
        try:
            _global_options(ctx=typer.Context(get_command(app)), read_only=True)
            assert is_read_only() is True
        finally:
            read_mode(False)
        assert is_read_only() is False

    def test_a_writer_under_the_flag_exits_three(self) -> None:
        """End to end through Typer, which is how a caller meets it."""
        from click.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(get_command(app), ["--read-only", "reembed", "--space", "bge-large"])
        assert result.exit_code == 3
        assert "read-only" in result.output

    def test_the_guard_raises_before_any_work(self) -> None:
        """The point of the mode: refused *before* the command does anything, so
        a half-applied write is not possible."""
        from app.cli import _refuse, read_mode  # noqa: PLC0415

        read_mode(True)
        try:
            with pytest.raises(typer.Exit) as caught:
                _refuse("ingest")
        finally:
            read_mode(False)
        assert caught.value.exit_code == 3

    def test_nothing_is_written_when_refused(self, capsys) -> None:  # noqa: ANN001
        from app.cli import _refuse, read_mode  # noqa: PLC0415

        read_mode(True)
        try:
            with pytest.raises(typer.Exit):
                _refuse("db:upgrade")
        finally:
            read_mode(False)
        out = capsys.readouterr().out
        assert "read-only" in out
        assert "Nothing was modified" in out

    def test_read_mode_off_is_the_default(self) -> None:
        from app.cli import is_read_only  # noqa: PLC0415

        assert is_read_only() is False


class TestMarking:
    def test_marking_twice_is_an_error(self) -> None:
        """Two markers on one command means one of them was meant for something
        else, and the second would be silently ignored."""
        from app.cli import writes  # noqa: PLC0415

        @writes("thing")
        def once() -> None:
            return None

        with pytest.raises(ValueError, match="already marked"):
            writes("thing")(once)
