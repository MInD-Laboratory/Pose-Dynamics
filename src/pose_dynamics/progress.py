"""Utility helpers for showing consistent progress bars in scripts and CLI commands."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")

_CONSOLE = Console()


def _progress_columns() -> tuple:
    return (
        SpinnerColumn(),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def run_steps(
    steps: Sequence[tuple[str, Callable[[], T]]],
    *,
    title: str = "Workflow",
    transient: bool = False,
) -> list[T]:
    """Execute callables sequentially while updating a shared progress bar."""
    if not steps:
        return []
    columns = _progress_columns()
    outputs: list[T] = []
    with Progress(*columns, console=_CONSOLE, transient=transient) as progress:
        task_id = progress.add_task(title, total=len(steps))
        for label, action in steps:
            progress.update(task_id, description=f"{title}: {label}")
            outputs.append(action())
            progress.advance(task_id)
    return outputs


def track_iterable(
    items: Sequence[T] | Iterable[T],
    *,
    title: str,
    handler: Callable[[T], None] | None = None,
    label_fn: Callable[[T], str] | None = None,
    transient: bool = False,
) -> None:
    """Iterate over items with a progress bar, optionally handling each item."""
    if isinstance(items, Sequence):
        total = len(items)
        iterator = iter(items)
    else:
        items = list(items)
        total = len(items)
        iterator = iter(items)
    if total == 0:
        return
    columns = _progress_columns()
    with Progress(*columns, console=_CONSOLE, transient=transient) as progress:
        task_id = progress.add_task(title, total=total)
        for item in iterator:
            label = label_fn(item) if label_fn else str(item)
            progress.update(task_id, description=f"{title}: {label}")
            if handler is not None:
                handler(item)
            progress.advance(task_id)


@contextmanager
def stage_progress(
    title: str,
    *,
    transient: bool = False,
) -> Iterator[Callable[[str], None]]:
    """Provide a single-task progress bar whose label can be updated on demand."""
    columns = _progress_columns()
    with Progress(*columns, console=_CONSOLE, transient=transient) as progress:
        task_id = progress.add_task(title, total=None)

        def _update(label: str) -> None:
            progress.update(task_id, description=f"{title}: {label}")

        try:
            yield _update
        finally:
            progress.update(task_id, description=f"{title}: complete")
            progress.stop_task(task_id)


@contextmanager
def stage_progress_with_total(
    title: str,
    total: int,
    *,
    transient: bool = False,
) -> Iterator[Callable[[str, int], None]]:
    """Similar to stage_progress but tracks a determinate total."""
    columns = _progress_columns()
    with Progress(*columns, console=_CONSOLE, transient=transient) as progress:
        task_id = progress.add_task(title, total=total)

        def _update(label: str, advance: int = 0) -> None:
            progress.update(task_id, description=f"{title}: {label}")
            if advance:
                progress.advance(task_id, advance)

        try:
            yield _update
        finally:
            progress.update(task_id, description=f"{title}: complete")
            progress.stop_task(task_id)
