"""APA-style tables rendered to HTML.

The formatting rules implemented here are the ones APA 7 actually specifies for
statistical tables, because a reviewer reading this instead of a PDF supplement
should not have to translate:

* **Rules.** Horizontal only, and only three of them -- above the header, below the
  header, and below the body. No vertical rules, no outer box, no zebra striping.
  Column spanners get their own rule spanning just the columns they cover.
* **Number and title.** Table number on its own line, then the title in italics.
* **Leading zeros.** Dropped for quantities that cannot exceed 1 (*p*, *r*), kept for
  everything else. :func:`fmt_p` and :func:`fmt_num` differ in exactly this.
* **Exact *p*.** Reported to three decimals, ``< .001`` below that. No asterisks --
  APA 7 prefers exact values, and a star column invites reading significance off a
  glyph rather than a number.
* **Notes.** Below the table, opening with an italic ``Note.``

Nothing here knows about recurrence, ROIs, or conditions: it renders any tidy frame.
The per-case-study assembly lives with the case study -- see
``examples/apa_report_case2.py`` for the pattern, which Cases 1 and 3 can follow by
supplying their own :class:`Table` list.
"""
from __future__ import annotations

import html
import inspect
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Table", "Section", "describe_by", "fmt_int", "fmt_num", "fmt_p", "fmt_signed",
    "render_report", "render_table",
]


# ----------------------------------------------------------------------
# Value formatting
# ----------------------------------------------------------------------
def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA


#: Typographic minus (U+2212), not the hyphen Python's formatter emits. In a column of
#: tabular figures a hyphen is visibly short and sits at the wrong height, and it is the
#: sign that carries the finding in a table of contrasts.
MINUS = "−"


def fmt_num(x, dp: int = 2) -> str:
    """Fixed-decimal number, leading zero kept. Use for quantities that can exceed 1."""
    if _missing(x):
        return "—"
    return f"{float(x):.{dp}f}".replace("-", MINUS)


def fmt_signed(x, dp: int = 2) -> str:
    """As :func:`fmt_num` but signed, for contrasts where direction is the point.

    A value that rounds to zero is rendered unsigned: ``+0.000`` would assert a
    direction the number does not actually resolve.
    """
    if _missing(x):
        return "—"
    v = float(x)
    if abs(round(v, dp)) == 0:
        return f"{0.0:.{dp}f}"
    return f"{v:+.{dp}f}".replace("-", MINUS)


def fmt_int(x) -> str:
    """Integer with thousands separators."""
    if _missing(x):
        return "—"
    return f"{int(round(float(x))):,}"


def fmt_p(x, dp: int = 3) -> str:
    """APA *p*: no leading zero, ``< .001`` at the floor, ``> .999`` at the ceiling.

    The floor is a reporting convention, not a rounding artifact -- a *p* of 0.0004 is
    reported as ``< .001`` rather than ``.000``, which would claim an impossible zero
    probability.
    """
    if _missing(x):
        return "—"
    p = float(x)
    if p < 0 or p > 1:
        raise ValueError(f"p-value outside [0, 1]: {p!r}")
    floor = 10.0 ** (-dp)
    if p < floor:
        return f"&lt; {('%.*f' % (dp, floor)).lstrip('0')}"
    if p > 1 - floor:
        return f"&gt; {('%.*f' % (dp, 1 - floor)).lstrip('0')}"
    return ("%.*f" % (dp, p)).lstrip("0")


def fmt_bounded(x, dp: int = 2) -> str:
    """Correlation-like value: no leading zero, sign preserved."""
    if _missing(x):
        return "—"
    s = "%.*f" % (dp, float(x))
    return s.replace("0.", ".").replace("-.", "−.") if s.startswith(("0.", "-0.")) else s


_NAMED: dict[str, Callable[[object], str]] = {
    "p": fmt_p,
    "int": fmt_int,
    "num": fmt_num,
    "num2": lambda v: fmt_num(v, 2),
    "num3": lambda v: fmt_num(v, 3),
    "num4": lambda v: fmt_num(v, 4),
    "signed": fmt_signed,
    "signed3": lambda v: fmt_signed(v, 3),
    "bounded": fmt_bounded,
    "text": lambda v: "—" if _missing(v) else str(v),
    # Cell values are HTML-escaped by default, which is right for anything derived from
    # data. Opt a column into markup with this when the values are labels you wrote --
    # subscripted measure names (L<sub>max</sub>) being the usual reason. Never apply it
    # to a column whose contents come from a file.
    "html": lambda v: "—" if _missing(v) else str(v),
}


def _resolve(fmt) -> Callable[[object], str]:
    if callable(fmt):
        return fmt
    if isinstance(fmt, str) and fmt in _NAMED:
        return _NAMED[fmt]
    raise ValueError(f"unknown formatter {fmt!r}; use a callable or one of {sorted(_NAMED)}")


def _takes_row(fn: Callable) -> bool:
    """Whether a formatter wants ``(value, row)`` rather than just ``(value)``.

    Row-aware formatters are what let one column hold measures of different magnitude
    -- a recurrence percentage near 30 and a correlation near 0.001 need different
    decimal places, and APA's fixed-decimal convention applies within a measure, not
    across a whole column.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):       # builtins without introspectable signatures
        return False
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len([p for p in positional if p.default is p.empty]) >= 2


# ----------------------------------------------------------------------
# Table model
# ----------------------------------------------------------------------
@dataclass
class Table:
    """One APA table.

    ``columns`` maps frame column -> header, in display order; headers may contain
    HTML so statistical symbols can be italicised (``"<i>p</i>"``). ``formatters``
    maps frame column -> a callable or a named formatter; anything unmapped is
    rendered as text. A formatter taking two arguments receives ``(value, row)``, so a
    column holding measures of different magnitude can vary its decimals per row --
    the frame may carry helper columns for that purpose which ``columns`` omits, and
    anything not named in ``columns`` is simply not rendered.

    ``spanners`` draws a grouped header row above the column headers as
    ``[(label, n_columns), ...]``, left to right, covering every column; use ``""``
    for a group that should stay unlabelled (typically the stub columns on the left).

    ``stub_groups`` names leading columns whose repeated consecutive values are
    blanked, so a stub reads as a group label rather than repeating down the page.
    """

    number: str
    title: str
    frame: pd.DataFrame
    columns: Mapping[str, str]
    formatters: Mapping[str, object] = field(default_factory=dict)
    note: str = ""
    spanners: Sequence[tuple[str, int]] = ()
    stub_groups: Sequence[str] = ()
    align: Mapping[str, str] = field(default_factory=dict)

    @property
    def anchor(self) -> str:
        return "table-" + str(self.number).replace(".", "-").replace(" ", "-").lower()


@dataclass
class Section:
    """A prose block between tables. ``body`` is raw HTML (already-formed ``<p>``)."""

    heading: str
    body: str = ""
    level: int = 2

    @property
    def anchor(self) -> str:
        return "sec-" + "".join(
            c if c.isalnum() else "-" for c in self.heading.lower()
        ).strip("-")


# ----------------------------------------------------------------------
# Data shaping
# ----------------------------------------------------------------------
def describe_by(frame: pd.DataFrame, metrics: Sequence[str], rows: Sequence[str],
                columns: str, column_order: Sequence[str] | None = None,
                dp: int = 3) -> tuple[pd.DataFrame, list[str], list[tuple[str, int]]]:
    """Cross-tabulate ``metrics`` as *M* (*SD*) cells, one column pair per level.

    Returns ``(frame, column_order, spanners)`` ready to hand to :class:`Table`: the
    frame carries one row per combination of ``rows`` plus ``metric``, and two columns
    per level of ``columns`` (``"{level} M"`` and ``"{level} SD"``).

    Descriptives are computed on whatever rows are passed, so aggregate to the level
    you intend to describe *before* calling -- window rows and trial means give
    different SDs, and the trial mean is usually the honest one because adjacent
    overlapping windows are not independent observations.
    """
    levels = list(column_order) if column_order is not None else list(
        pd.unique(frame[columns].dropna()))
    long = frame.melt(id_vars=[*rows, columns], value_vars=list(metrics),
                      var_name="metric", value_name="_v")
    agg = (long.dropna(subset=["_v"])
           .groupby([*rows, "metric", columns], observed=True)["_v"]
           .agg(["mean", "std"]).reset_index())

    out = agg.pivot_table(index=[*rows, "metric"], columns=columns,
                          values=["mean", "std"], observed=True)
    ordered: dict[str, list] = {}
    for lvl in levels:
        for stat in ("mean", "std"):
            key = (stat, lvl)
            ordered[f"{lvl} {'M' if stat == 'mean' else 'SD'}"] = (
                out[key] if key in out.columns else np.nan)
    flat = pd.DataFrame(ordered, index=out.index).reset_index()

    # Preserve the caller's row order rather than pandas' lexical sort.
    for col, order in ((rows[0], None), ("metric", list(metrics))):
        if order is not None:
            flat["metric"] = pd.Categorical(flat["metric"], categories=order, ordered=True)
    flat = flat.sort_values([*rows, "metric"]).reset_index(drop=True)

    col_order = [*rows, "metric", *ordered.keys()]
    spanners = [("", len(rows) + 1)] + [(lvl, 2) for lvl in levels]
    return flat[col_order], col_order, spanners


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def _cell(value, fmt, row=None) -> str:
    if fmt is None:
        return "—" if _missing(value) else html.escape(str(value))
    fn = _resolve(fmt)
    return fn(value, row) if _takes_row(fn) else fn(value)


def render_table(table: Table) -> str:
    """One ``<figure class="apa">`` block: number, italic title, table, note."""
    cols = list(table.columns)
    align = {c: table.align.get(c, "right") for c in cols}
    for c in cols:
        if c not in table.align and table.frame[c].dtype == object:
            align[c] = "left"

    head = []
    if table.spanners:
        cells, i = [], 0
        for label, span in table.spanners:
            klass = "spanner" if label else "spanner-blank"
            cells.append(
                f'<th class="{klass}" colspan="{span}" scope="colgroup">'
                f"{label}</th>")
            i += span
        if i != len(cols):
            raise ValueError(
                f"spanners cover {i} columns but the table has {len(cols)}")
        head.append("<tr>" + "".join(cells) + "</tr>")
    head.append("<tr>" + "".join(
        f'<th class="a-{align[c]}" scope="col">{table.columns[c]}</th>' for c in cols
    ) + "</tr>")

    body, previous = [], {}
    for _, row in table.frame.iterrows():
        cells = []
        carry = False
        for c in cols:
            text = _cell(row[c], table.formatters.get(c), row)
            if c in table.stub_groups:
                if not carry and previous.get(c) == text:
                    text = ""
                else:
                    carry = True
                    previous[c] = text
            cells.append(f'<td class="a-{align[c]}">{text}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    note = (f'<figcaption class="note"><i>Note.</i> {table.note}</figcaption>'
            if table.note else "")
    return (
        f'<figure class="apa" id="{table.anchor}">\n'
        f'  <div class="tnum">Table {html.escape(str(table.number))}</div>\n'
        f'  <div class="ttitle">{table.title}</div>\n'
        f'  <div class="scroll">\n'
        f'    <table>\n      <thead>{"".join(head)}</thead>\n'
        f'      <tbody>{"".join(body)}</tbody>\n    </table>\n'
        f"  </div>\n{note}\n</figure>"
    )


_CSS = """
:root{
  --ground:#ffffff; --raised:#f7f9fb; --ink:#16202b; --muted:#5c6b7a;
  --rule:#c3ced9; --hair:#e2e8ee; --accent:#20517e; --wash:#eef4fa;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#11171d; --raised:#171f27; --ink:#e6edf4; --muted:#9aabbb;
    --rule:#3a4753; --hair:#232d37; --accent:#93b8db; --wash:#18222c;
  }
}
:root[data-theme="dark"]{
  --ground:#11171d; --raised:#171f27; --ink:#e6edf4; --muted:#9aabbb;
  --rule:#3a4753; --hair:#232d37; --accent:#93b8db; --wash:#18222c;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.62;
  -webkit-text-size-adjust:100%;
}
.wrap{max-width:min(1180px,94vw); margin:0 auto; padding:4rem 0 6rem}
.measure{max-width:68ch}
header.doc{border-bottom:2px solid var(--ink); padding-bottom:1.6rem; margin-bottom:2.4rem}
.eyebrow{
  font-family:var(--sans); font-size:.72rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .9rem
}
h1{font-size:2.05rem; line-height:1.2; margin:0 0 .7rem; font-weight:600; text-wrap:balance}
.standfirst{margin:0; color:var(--muted); font-size:1.03rem; max-width:64ch}
h2{
  font-size:1.32rem; font-weight:600; margin:3.4rem 0 .9rem; text-wrap:balance;
  padding-bottom:.4rem; border-bottom:1px solid var(--hair)
}
h3{font-size:1.06rem; font-weight:600; margin:2.2rem 0 .6rem}
p{margin:0 0 1rem; max-width:68ch}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:3px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.87em;
  background:var(--wash); padding:.1em .35em; border-radius:3px}

nav.toc{background:var(--raised); border:1px solid var(--hair); border-radius:4px;
  padding:1.3rem 1.5rem; margin:0 0 2.6rem}
nav.toc h2{font-size:.72rem; font-family:var(--sans); font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 .8rem; border:0; padding:0}
nav.toc ol{margin:0; padding:0; list-style:none; display:grid; gap:.42rem}
nav.toc a{text-decoration:none; display:flex; gap:.7rem; align-items:baseline}
nav.toc a:hover{text-decoration:underline}
nav.toc .n{font-family:var(--sans); font-size:.78rem; font-weight:600; color:var(--muted);
  min-width:4.6em; font-variant-numeric:tabular-nums}
nav.toc .t{color:var(--ink)}

figure.apa{margin:2.6rem 0 3rem}
.tnum{font-family:var(--sans); font-size:.78rem; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--accent)}
.ttitle{font-style:italic; margin:.15rem 0 .8rem; max-width:72ch}
.scroll{overflow-x:auto; padding-bottom:.2rem}
table{border-collapse:collapse; width:100%; font-family:var(--sans); font-size:.845rem;
  font-variant-numeric:tabular-nums; line-height:1.45}
thead tr:first-child th{border-top:1.4px solid var(--ink)}
thead tr:last-child th{border-bottom:1.1px solid var(--ink)}
th,td{padding:.42rem .7rem; vertical-align:baseline; white-space:nowrap}
th{font-weight:600; text-align:right}
th.spanner{text-align:center; border-bottom:1px solid var(--rule); padding-bottom:.2rem;
  font-weight:600}
th.spanner-blank{border:0}
thead tr:first-child th.spanner-blank{border-top:1.4px solid var(--ink)}
tbody tr:last-child td{border-bottom:1.1px solid var(--ink)}
td{color:var(--ink)}
.a-left{text-align:left} .a-right{text-align:right} .a-center{text-align:center}
tbody td.a-left{white-space:normal}
.note{font-size:.83rem; color:var(--muted); margin-top:.7rem; max-width:76ch;
  line-height:1.55}
.note i{font-style:italic}

.meta{display:grid; gap:.35rem; font-family:var(--sans); font-size:.82rem;
  color:var(--muted); margin:1.4rem 0 0}
.meta div{display:flex; gap:.6rem}
.meta dt{font-weight:600; min-width:9.5em; color:var(--ink)}
footer.doc{margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--hair);
  font-family:var(--sans); font-size:.8rem; color:var(--muted)}
@media (max-width:640px){
  body{font-size:16px} .wrap{padding:2.4rem 0 4rem} h1{font-size:1.62rem}
  nav.toc a{flex-direction:column; gap:.1rem}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def render_report(title: str, blocks: Iterable[Table | Section], *,
                  eyebrow: str = "", standfirst: str = "", meta: Mapping[str, str] | None = None,
                  footer: str = "", standalone: bool = True, toc: bool = True) -> str:
    """Assemble ``blocks`` into one document.

    ``standalone`` wraps the result in a full HTML5 document so the file opens on its
    own; pass ``False`` for a fragment to embed somewhere that already supplies
    ``<head>``. ``toc`` prepends an index of the tables -- useful when a reader arrives
    looking for one specific table, unwanted when the page is meant to be nothing but
    the tables themselves.
    """
    blocks = list(blocks)
    toc_items, body = [], []
    for b in blocks:
        if isinstance(b, Table):
            toc_items.append(f'<li><a href="#{b.anchor}"><span class="n">Table '
                             f'{html.escape(str(b.number))}</span>'
                             f'<span class="t">{b.title}</span></a></li>')
            body.append(render_table(b))
        else:
            body.append(f'<h{b.level} id="{b.anchor}">{b.heading}</h{b.level}>')
            if b.body:
                body.append(b.body)

    meta_html = ""
    if meta:
        rows = "".join(f"<div><dt>{html.escape(k)}</dt><dd>{v}</dd></div>"
                       for k, v in meta.items())
        meta_html = f'<dl class="meta">{rows}</dl>'

    inner = (
        '<div class="wrap">\n'
        '<header class="doc">'
        + (f'<p class="eyebrow">{html.escape(eyebrow)}</p>' if eyebrow else "")
        + f"<h1>{title}</h1>"
        + (f'<p class="standfirst">{standfirst}</p>' if standfirst else "")
        + meta_html
        + "</header>\n"
        + (f'<nav class="toc"><h2>Tables</h2><ol>{"".join(toc_items)}</ol></nav>\n'
           if toc and toc_items else "")
        + "\n".join(body)
        + (f'<footer class="doc">{footer}</footer>' if footer else "")
        + "\n</div>"
    )
    if not standalone:
        return f"<style>{_CSS}</style>\n{inner}"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{html.escape(_strip_tags(title))}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{inner}\n</body>\n</html>\n"
    )


def _strip_tags(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)
