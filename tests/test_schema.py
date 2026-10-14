import copy

import pytest

from evalkit.schema import load_dataset, validate_dataset

GOOD = {"dataset_version": "1.0.0", "cases": [{"id": "a", "question": "Q?"}]}


def test_bundled_dataset_is_valid():
    version, cases = load_dataset("data/cases_v1.json")
    assert version == "1.0.0"
    assert len(cases) >= 10


def test_minimal_case_gets_defaults():
    _, cases = validate_dataset(GOOD)
    assert cases[0].must_include_any == [] and cases[0].require_disclaimer is False


def test_duplicate_ids_rejected():
    bad = copy.deepcopy(GOOD)
    bad["cases"].append({"id": "a", "question": "Another?"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(bad)


def test_unknown_key_rejected_so_typos_are_caught():
    bad = copy.deepcopy(GOOD)
    bad["cases"][0]["must_not_includes"] = ["x"]  # typo of must_not_include
    with pytest.raises(ValueError, match="unknown keys"):
        validate_dataset(bad)


@pytest.mark.parametrize("patch", [
    {"question": ""},
    {"max_words": 0},
    {"max_words": True},
    {"require_disclaimer": "yes"},
    {"tags": "lifestyle"},
])
def test_invalid_field_values_rejected(patch):
    bad = copy.deepcopy(GOOD)
    bad["cases"][0].update(patch)
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_missing_version_rejected():
    with pytest.raises(ValueError, match="dataset_version"):
        validate_dataset({"cases": GOOD["cases"]})
