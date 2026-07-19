"""Gate 3 normalization — offline, deterministic. The single biggest silent-failure guard:
if these break, every downstream lookup misses without erroring."""
from __future__ import annotations

import pytest

from oncoreason import variants as V


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("p.L858R", "L858R"),
        ("L858R", "L858R"),
        ("p.Leu858Arg", "L858R"),
        ("Leu858Arg", "L858R"),
        ("p.(Leu858Arg)", "L858R"),
        ("p.G12C", "G12C"),
        ("p.Gly12Cys", "G12C"),
        ("p.V600E", "V600E"),
        ("p.T790M", "T790M"),
    ],
)
def test_all_spellings_reconcile(raw, expected):
    assert V.civic_variant_name(raw) == expected


def test_profile_name():
    assert V.civic_profile_name("egfr", "p.Leu858Arg") == "EGFR L858R"


def test_non_substitution_preserved_not_mangled():
    # three-letter codes convert, but the delins structure survives
    assert V.three_to_one("p.Leu747_Pro753delinsSer") == "L747_P753delinsS"
    # a free-text event is passed through untouched
    assert V.three_to_one("exon 19 deletion") == "exon 19 deletion"
    assert V.is_substitution("exon 19 deletion") is False
    assert V.is_substitution("p.L858R") is True


def test_round_trip_to_three_letter_for_clinvar():
    assert V.one_to_three("p.L858R") == "p.Leu858Arg"
    assert V.one_to_three("G12C") == "p.Gly12Cys"
    # stop codon
    assert V.one_to_three("R213*") == "p.Arg213Ter"
    # non-substitution -> None (caller falls back, doesn't guess)
    assert V.one_to_three("exon 19 deletion") is None
