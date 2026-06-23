"""Pure unit tests for the R-ADD adaptive restraint-window insertion engine.

These import the pure R-ADD helpers from ``implicit_solvent_ddm.adaptive_restraints`` and
deliberately do NOT touch the Toil/MBAR ``run_workflow`` session fixture in ``conftest.py`` (which
runs the full DDM workflow). Run with::

    pytest implicit_solvent_ddm/tests/test_adaptive_insertion.py -v
"""
import pytest

from implicit_solvent_ddm.adaptive_restraints import (
    derive_offset,
    min_direction_superdiagonal,
    find_bad_sections,
    select_section,
    worst_gap_index,
    snap_to_pool,
    insert_one,
)

THRESH = 0.04


# ---------------------------------------------------------------------------
# T8 — OFFSET derived & asserted (con/orient pairing is the atomic unit)
# ---------------------------------------------------------------------------
def test_derive_offset_constant():
    con = [-8.0, -2.0, 4.0]
    orient = [-4.0, 2.0, 8.0]
    assert derive_offset(con, orient) == 4.0


def test_derive_offset_nonconstant_raises():
    with pytest.raises(ValueError):
        derive_offset([-8.0, -2.0, 4.0], [-4.0, 2.0, 9.0])


def test_derive_offset_length_mismatch_raises():
    with pytest.raises(ValueError):
        derive_offset([-8.0, -2.0], [-4.0, 2.0, 8.0])


# ---------------------------------------------------------------------------
# T6 — min-direction superdiagonal: a one-sided weak pair registers as weak
# ---------------------------------------------------------------------------
def test_min_direction_superdiagonal_uses_min_not_average():
    # pair (0,1) is one-sided weak: fwd=0.01, rev=0.07 -> symmetric average 0.04 would PASS,
    # but the min-direction value 0.01 is correctly below threshold.
    O = [
        [0.90, 0.01, 0.00],
        [0.07, 0.85, 0.06],
        [0.00, 0.06, 0.94],
    ]
    sd = min_direction_superdiagonal(O)
    assert sd == [0.01, 0.06]
    assert sd[0] < THRESH


# ---------------------------------------------------------------------------
# T1 — bad-section finding + selection tie-breaks (total order)
# ---------------------------------------------------------------------------
def test_find_bad_sections():
    sd = [0.06, 0.02, 0.01, 0.08, 0.03]
    assert find_bad_sections(sd, THRESH) == [[1, 2], [4]]


def test_find_bad_sections_none():
    assert find_bad_sections([0.06, 0.08, 0.10], THRESH) == []


def test_select_section_longest_wins():
    sd = [0.02, 0.01, 0.08, 0.03]  # sections [[0, 1], [3]]
    assert select_section([[0, 1], [3]], sd) == [0, 1]


def test_select_section_tiebreak_min_overlap():
    # equal-length sections -> the one with the smaller MINIMUM overlap wins
    sd = [0.03, 0.08, 0.005]  # [[0]] (min 0.03) vs [[2]] (min 0.005)
    assert select_section([[0], [2]], sd) == [2]


def test_select_section_tiebreak_start_index():
    # equal length, equal minimum -> lowest start index
    sd = [0.01, 0.08, 0.01]  # [[0]] and [[2]], both min 0.01
    assert select_section([[0], [2]], sd) == [0]


# ---------------------------------------------------------------------------
# T2 — worst sub-gap within a section
# ---------------------------------------------------------------------------
def test_worst_gap_index():
    sd = [0.02, 0.005, 0.03]
    assert worst_gap_index([0, 1, 2], sd) == 1


def test_worst_gap_index_ties_low():
    sd = [0.01, 0.05, 0.01]
    assert worst_gap_index([0, 2], sd) == 0


# ---------------------------------------------------------------------------
# T3 — snap to nearest free pool candidate; ties -> lower exponent
# ---------------------------------------------------------------------------
def test_snap_to_pool_nearest():
    pool = [-7.0, -6.0, -5.0, -4.0, -3.0]
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected=set()) == -5.0


def test_snap_to_pool_tie_breaks_low():
    pool = [-6.0, -4.0]  # both equidistant from ideal -5 -> choose the lower (-6)
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected=set()) == -6.0


def test_snap_to_pool_skips_selected_and_outside_gap():
    pool = [-9.0, -5.0, -2.0, 5.0]  # -9, -2, 5 lie outside the open gap (-8, -2)
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected={-5.0}) is None


# ---------------------------------------------------------------------------
# T2 (one full step) — insert at the log2 midpoint of the single worst gap
# ---------------------------------------------------------------------------
def test_insert_one_picks_worst_gap_midpoint():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.06]  # pair (-8,-2) weak; pair (-2,4) ok
    pool = [-7.0, -6.0, -5.0, -4.0, -3.0]
    new_con, converged, reason = insert_one(
        block, sd, pool, THRESH, lower_bound=-8.0, upper_bound=4.0
    )
    assert converged is False
    assert new_con == -5.0  # midpoint of (-8,-2), snapped to the matching pool point


def test_insert_one_targets_the_worst_of_several_gaps():
    block = [-8.0, -4.0, 0.0, 4.0]
    sd = [0.03, 0.10, 0.005]  # weak: pair0 (0.03) and pair2 (0.005); pair2 is worse
    pool = [-6.0, -2.0, 2.0]
    new_con, converged, _ = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert converged is False
    assert new_con == 2.0  # midpoint of the worst gap (0,4)


# ---------------------------------------------------------------------------
# T1 — converged when every adjacent overlap is adequate
# ---------------------------------------------------------------------------
def test_insert_one_converged_when_all_overlaps_adequate():
    block = [-8.0, -2.0, 4.0]
    sd = [0.08, 0.06]
    assert insert_one(block, sd, [-5.0], THRESH, -8.0, 4.0) == (None, True, "converged")


# ---------------------------------------------------------------------------
# T4 — anchor protection: never returns con at/below endstate or at/above max
# ---------------------------------------------------------------------------
def test_insert_one_never_breaches_anchors():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.01]  # both gaps weak
    # pool offers only candidates at/below the endstate or at/above the pinned max
    pool = [-9.0, -8.0, 4.0, 5.0]
    new_con, converged, reason = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert new_con is None
    assert converged is True
    assert reason == "pool-exhausted"


# ---------------------------------------------------------------------------
# T5 — pool exhaustion terminates (no infinite recursion) and is distinguished
#      from clean convergence
# ---------------------------------------------------------------------------
def test_insert_one_pool_exhausted_terminates():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.02]  # both gaps weak
    new_con, converged, reason = insert_one(block, sd, [], THRESH, -8.0, 4.0)
    assert new_con is None and converged is True and reason == "pool-exhausted"


def test_insert_one_partial_fill_then_pool_exhausted():
    # gap0 (-8,-2) is fillable from the pool; gap1 (-2,4) is not -> first call fills gap0.
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.01]
    pool = [-5.0]  # only a candidate for gap0
    new_con, converged, reason = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert new_con == -5.0 and converged is False  # worst-tie -> lowest k=0 -> gap0 filled first


def test_insert_one_length_mismatch_raises():
    with pytest.raises(ValueError):
        insert_one([-8.0, -2.0, 4.0], [0.01], [-5.0], THRESH, -8.0, 4.0)
