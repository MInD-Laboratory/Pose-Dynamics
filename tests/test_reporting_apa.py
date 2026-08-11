"""
APA reporting: the formatting conventions, not the prose.

These tests pin the rules a reader relies on to interpret a table without being told
them -- a dropped leading zero means the quantity is bounded by 1, ``< .001`` means a
floor rather than a measured zero, and a blank stub means "same as above" rather than
missing. Each is a silent correctness bug if it regresses: the number still renders,
it just says something different from what it means.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pose_dynamics.reporting import Table, describe_by, fmt_num, fmt_p, fmt_signed
from pose_dynamics.reporting.apa import MINUS, Section, render_report, render_table


def _text_rows(html: str) -> list[list[str]]:
    body = re.search(r"<tbody>(.*?)</tbody>", html, re.S).group(1)
    return [[re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row)]
            for row in re.findall(r"<tr>(.*?)</tr>", body, re.S)]


# ----------------------------------------------------------------------
# Value formatting
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (0.0004, "&lt; .001"),      # floor, not ".000" -- an impossible zero probability
    (0.001, ".001"),
    (0.0231, ".023"),           # leading zero dropped: p cannot exceed 1
    (0.5, ".500"),
    (0.9999, "&gt; .999"),
    (1.0, "&gt; .999"),
])
def test_fmt_p_conventions(value, expected):
    assert fmt_p(value) == expected


def test_fmt_p_rejects_impossible_values():
    # A p outside [0, 1] means the caller handed the wrong column, and silently
    # formatting it would hide that in a table of otherwise plausible numbers.
    with pytest.raises(ValueError):
        fmt_p(1.4)


def test_fmt_num_keeps_leading_zero():
    # Unlike p, these quantities can exceed 1, so the zero carries information.
    assert fmt_num(0.0594, 4) == "0.0594"
    assert fmt_num(24.281, 2) == "24.28"


def test_negatives_use_typographic_minus():
    assert fmt_num(-0.0065, 4) == f"{MINUS}0.0065"
    assert fmt_signed(-0.0121, 4) == f"{MINUS}0.0121"
    assert "-" not in fmt_signed(-1.5, 2)


def test_fmt_signed_does_not_sign_a_rounded_zero():
    # "+0.0000" asserts a direction the number does not resolve.
    assert fmt_signed(0.00001, 4) == "0.0000"
    assert fmt_signed(-0.00001, 4) == "0.0000"
    assert fmt_signed(0.002, 3) == "+0.002"


def test_missing_renders_as_em_dash():
    for fn in (fmt_num, fmt_signed, fmt_p):
        assert fn(np.nan) == "—"
        assert fn(None) == "—"


# ----------------------------------------------------------------------
# Table rendering
# ----------------------------------------------------------------------
@pytest.fixture
def frame():
    return pd.DataFrame({
        "roi": ["Arms", "Arms", "Upper body"],
        "measure": ["%REC", "%DET", "%REC"],
        "b": [0.0231, -1.5, np.nan],
        "p": [0.00004, 0.5, 0.2],
    })


def _table(frame, **kw):
    return Table(number="1", title="T", frame=frame,
                 columns={"roi": "ROI", "measure": "Measure", "b": "<i>b</i>",
                          "p": "<i>p</i>"},
                 formatters={"b": "signed3", "p": "p"}, **kw)


def test_stub_groups_blank_repeats_only_when_consecutive(frame):
    rows = _text_rows(render_table(_table(frame, stub_groups=["roi"])))
    assert [r[0] for r in rows] == ["Arms", "", "Upper body"]


def test_stub_group_repeats_after_an_intervening_value():
    # A value returning after a different one must reappear, or the table claims the
    # rows belong to whichever group last printed.
    f = pd.DataFrame({"roi": ["A", "B", "A"], "measure": list("xyz"),
                      "b": [1.0, 2.0, 3.0], "p": [0.1, 0.2, 0.3]})
    rows = _text_rows(render_table(_table(f, stub_groups=["roi"])))
    assert [r[0] for r in rows] == ["A", "B", "A"]


def test_row_aware_formatter_varies_precision_per_row(frame):
    dp = {"%REC": 3, "%DET": 1}
    t = _table(frame)
    t.formatters = {**t.formatters,
                    "b": lambda v, row: fmt_signed(v, dp[row["measure"]])}
    rows = _text_rows(render_table(t))
    assert rows[0][2] == "+0.023"
    assert rows[1][2] == f"{MINUS}1.5"


def test_spanner_column_count_is_validated(frame):
    with pytest.raises(ValueError, match="spanners cover"):
        render_table(_table(frame, spanners=[("", 2), ("Group", 1)]))   # 3 of 4


def test_no_vertical_rules_or_outer_box(frame):
    # APA tables carry three horizontal rules and nothing else; a border-left or a
    # full outline is the most common way a generated table stops looking like one.
    html = render_table(_table(frame))
    assert "border-left" not in html and "border-right" not in html


def test_html_in_headers_survives_but_data_is_escaped():
    f = pd.DataFrame({"roi": ["<script>x</script>"], "b": [1.0]})
    html = render_table(Table(number="1", title="T", frame=f,
                              columns={"roi": "ROI", "b": "<i>b</i>"}))
    assert "<i>b</i>" in html
    assert "<script>" not in html


def test_html_formatter_opts_a_label_column_out_of_escaping():
    # Measure names carry markup a reader is meant to see -- L_max is a subscript, not
    # the literal text "L<sub>max</sub>" -- so a column of authored labels can opt in.
    f = pd.DataFrame({"measure": ["L<sub>max</sub>"], "b": [1.0]})
    html = render_table(Table(number="1", title="T", frame=f,
                              columns={"measure": "Measure", "b": "<i>b</i>"},
                              formatters={"measure": "html"}))
    assert "L<sub>max</sub>" in html and "&lt;sub&gt;" not in html


def test_escaping_is_the_default_for_unmapped_columns():
    f = pd.DataFrame({"measure": ["L<sub>max</sub>"], "b": [1.0]})
    html = render_table(Table(number="1", title="T", frame=f,
                              columns={"measure": "Measure", "b": "<i>b</i>"}))
    assert "&lt;sub&gt;" in html


# ----------------------------------------------------------------------
# describe_by
# ----------------------------------------------------------------------
def test_describe_by_pairs_columns_per_level_in_requested_order():
    f = pd.DataFrame({
        "roi": ["A"] * 4, "condition": ["Loud", "Loud", "Quiet", "Quiet"],
        "m1": [1.0, 3.0, 10.0, 12.0],
    })
    out, cols, spanners = describe_by(f, ["m1"], rows=["roi"], columns="condition",
                                      column_order=["Quiet", "Loud"])
    assert cols == ["roi", "metric", "Quiet M", "Quiet SD", "Loud M", "Loud SD"]
    assert spanners == [("", 2), ("Quiet", 2), ("Loud", 2)]
    assert out.loc[0, "Quiet M"] == pytest.approx(11.0)
    assert out.loc[0, "Loud M"] == pytest.approx(2.0)


def test_describe_by_tolerates_a_level_with_no_rows():
    f = pd.DataFrame({"roi": ["A", "A"], "condition": ["Quiet", "Quiet"], "m1": [1.0, 3.0]})
    out, _, _ = describe_by(f, ["m1"], rows=["roi"], columns="condition",
                            column_order=["Quiet", "Loud"])
    assert np.isnan(out.loc[0, "Loud M"])


# ----------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------
def test_report_indexes_tables_and_links_to_them(frame):
    html = render_report("Title", [Section("S", "<p>x</p>"), _table(frame)])
    t = _table(frame)
    assert f'href="#{t.anchor}"' in html
    assert f'id="{t.anchor}"' in html


def test_standalone_and_fragment_differ_only_in_the_wrapper(frame):
    blocks = [_table(frame)]
    full = render_report("T", blocks, standalone=True)
    frag = render_report("T", blocks, standalone=False)
    assert full.startswith("<!doctype html>") and "<title>" in full
    assert not frag.lstrip().startswith("<!doctype")
    # `<header>` legitimately appears in both, so match the document tags exactly.
    for tag in ("<head>", "<body>", "<html"):
        assert tag not in frag
    assert "<figure class=\"apa\"" in frag


def test_every_colour_token_is_defined_on_bare_root(frame):
    """The un-stamped system-theme state must resolve every token.

    A token defined only inside ``@media (prefers-color-scheme: dark)`` or
    ``[data-theme=...]`` is invisible to a viewer on the default "system" setting,
    which renders one theme's text on the other theme's ground.
    """
    css = re.search(r"<style>(.*?)</style>", render_report("T", [_table(frame)]), re.S).group(1)
    declared = set(re.findall(r"--([a-z-]+)\s*:", re.search(r":root\{(.*?)\}", css, re.S).group(1)))
    assert not set(re.findall(r"var\(--([a-z-]+)\)", css)) - declared
