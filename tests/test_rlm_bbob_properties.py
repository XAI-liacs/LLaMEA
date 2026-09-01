import pytest

from llamea.rlm_surrogate.bbob_properties import BBOB_FUNCTIONS, describe_bbob_function


def test_all_24_functions_present():
    assert set(BBOB_FUNCTIONS) == set(range(1, 25))


def test_describe_sphere_separable_unimodal():
    text = describe_bbob_function(1)
    assert "Sphere" in text
    assert "separable=True" in text
    assert "unimodal" in text
    assert "group 1" in text


def test_describe_lunacek_multimodal_weak_structure():
    text = describe_bbob_function(24)
    assert "Lunacek" in text
    assert "separable=False" in text
    assert "multi-modal" in text
    assert "group 5" in text


def test_describe_unknown_fid_raises():
    with pytest.raises(KeyError):
        describe_bbob_function(25)


def test_group_names_cover_all_groups_used():
    groups_used = {info.group for info in BBOB_FUNCTIONS.values()}
    assert groups_used == {1, 2, 3, 4, 5}
