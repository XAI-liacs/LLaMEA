import pytest

from llamea import (
    VARIANTS,
    LLaMEA,
    LLaMEA_BO,
    LLaMEA_HPO,
    LLaMEA_MO,
    LLaMEA_SAGE,
    get_variant,
)
from llamea.llm import LLM
from llamea.multi_objective_fitness import Fitness


class FakeLLM(LLM):
    def __init__(self):
        super().__init__("", "fake")

    def query(self, session_messages):
        return (
            "# Description: X\n# Code:\n```python\nclass X:\n    pass\n```\n"
            "Space: ```python\n{'a': (0.0, 1.0)}\n```"
        )


def f(solution, logger=None):
    solution.set_scores(1.0, "ok")
    return solution


def test_hpo_preset():
    opt = LLaMEA_HPO(f, FakeLLM(), log=False)
    assert opt.HPO is True
    assert (opt.n_parents, opt.n_offspring) == (1, 1)
    assert "Space" in opt.output_format_prompt
    assert "redesign" in opt.operators[0].prompt


def test_bo_preset_and_override():
    opt = LLaMEA_BO(f, FakeLLM(), log=False)
    assert (opt.n_parents, opt.n_offspring, opt.budget) == (4, 16, 100)
    assert "Bayesian" in opt.task_prompt
    names = {o.name: o for o in opt.operators}
    assert names["crossover"].number_of_parents == 2
    assert names["crossover"].weight == pytest.approx(0.6)
    # user arguments win over the preset
    opt = LLaMEA_BO(f, FakeLLM(), budget=7, n_offspring=3, log=False)
    assert (opt.budget, opt.n_offspring) == (7, 3)


def test_presets_are_not_shared_between_instances():
    a = LLaMEA_BO(f, FakeLLM(), log=False)
    b = LLaMEA_BO(f, FakeLLM(), log=False)
    assert a.operators[0].id != b.operators[0].id


def test_mo_requires_objectives_and_runs():
    with pytest.raises(ValueError):
        LLaMEA_MO(f, FakeLLM(), task_prompt="t", log=False)

    def mo_f(solution, logger=None):
        solution.set_scores(Fitness({"a": 1.0, "b": 2.0}), "ok")
        return solution

    opt = LLaMEA_MO(
        mo_f,
        FakeLLM(),
        objectives=["a", "b"],
        task_prompt="t",
        n_parents=2,
        n_offspring=2,
        budget=4,
        max_workers=1,
        log=False,
    )
    assert opt.multi_objective and opt.multi_objective_keys == ["a", "b"]
    front = opt.run()
    assert isinstance(front, list) and front


def test_mo_warns_without_task_prompt():
    with pytest.warns(UserWarning):
        LLaMEA_MO(f, FakeLLM(), objectives=["a"], log=False)


def test_sage_preset():
    pytest.importorskip("xgboost")
    assert LLaMEA_SAGE(f, FakeLLM(), log=False).feature_guided_mutation


def test_registry_lookup():
    assert get_variant("bo") is LLaMEA_BO
    assert get_variant("LLaMEA-HPO") is LLaMEA_HPO
    assert get_variant("llamea_mo") is LLaMEA_MO
    assert get_variant("llamea") is LLaMEA
    assert set(VARIANTS) == {"llamea", "hpo", "bo", "mo", "sage"}
    with pytest.raises(KeyError):
        get_variant("nope")


def test_variants_are_llamea_and_have_papers():
    for cls in (LLaMEA_HPO, LLaMEA_BO, LLaMEA_SAGE):
        assert issubclass(cls, LLaMEA)
        assert cls.paper_url.startswith("https://")
