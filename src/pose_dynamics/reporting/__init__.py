"""Report rendering: APA-style tables and standalone HTML documents.

Deliberately not imported by the package root. Nothing here is part of the analysis
pipeline -- it turns tidy result frames into something a reader (or a reviewer) can
read, and it is the case-study notebooks and scripts that call it, never the
preprocessing or recurrence code.

    from pose_dynamics.reporting import Table, render_report

See :mod:`pose_dynamics.reporting.apa` for the formatting rules it applies.
"""
from .apa import (
    Table,
    describe_by,
    fmt_bounded,
    fmt_int,
    fmt_num,
    fmt_p,
    fmt_signed,
    render_report,
    render_table,
)

__all__ = [
    "Table",
    "describe_by",
    "fmt_bounded",
    "fmt_int",
    "fmt_num",
    "fmt_p",
    "fmt_signed",
    "render_report",
    "render_table",
]
