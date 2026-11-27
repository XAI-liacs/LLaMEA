"""Regression tests for the bugs fixed in LLaMEA 1.6.0."""

import copy
import pickle

import pytest
from joblib import delayed

from llamea import Dummy_LLM, LLaMEA, Operator, Solution
from llamea.llm import LLM
from llamea.multi_objective_fitness import Fitness
from llamea.utils import NoCodeException, prepare_namespace
from llamea.weight_updaters import DefaultWeightUpdater, DiscountedUCBState


class ScriptedLLM(LLM):
    """Returns a fixed response; no network access."""

    def __init__(self, response):
        super().__init__("", "scripted/model:1")
        self.response = response

    def query(self, session_messages):
        return self.response


GOOD = (
    "# Description: Tiny\n# Code:\n```python\nclass Tiny:\n"
    "    def __init__(self, budget=1, dim=1):\n        pass\n```"
)


def scalar_f(solution, logger=None):
    solution.set_scores(float(len(solution.code)), "ok")
    return solution


def make(fitness, code="a=1"):
    s = Solution(code=code)
    s.set_scores(fitness)
    return s


# ---------------------------------------------------------------- core loop


def test_operator_weight_updater_is_used_and_updated():
    ops = [Operator("mutate", weight=1.0), Operator("redesign", weight=1.0)]
    updater = DiscountedUCBState([o.id for o in ops])
    opt = LLaMEA(
        scalar_f,
        ScriptedLLM(GOOD),
        n_parents=1,
        n_offspring=2,
        operators=ops,
        operator_weight_updater=updater,
        budget=5,
        max_workers=1,
        log=False,
    )
    assert opt.operator_weight_updater is updater
    opt.run()
    # Every generated offspring was rewarded.
    assert sum(updater.sum_weights.values()) > 0


def test_default_keeps_configured_operator_weights():
    ops = [Operator("a", weight=0.6), Operator("b", weight=0.4)]
    opt = LLaMEA(
        scalar_f,
        ScriptedLLM(GOOD),
        n_parents=1,
        n_offspring=1,
        operators=ops,
        budget=4,
        max_workers=1,
        log=False,
    )
    opt.run()
    assert [o.weight for o in opt.operators] == [0.6, 0.4]


def test_old_archives_with_default_updater_keep_weights():
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), log=False)
    state = opt.__dict__.copy()
    state["operator_weight_updater"] = DefaultWeightUpdater(
        [o.id for o in opt.operators]
    )
    restored = LLaMEA.__new__(LLaMEA)
    restored.__setstate__(state)
    assert restored.operator_weight_updater is None


def test_operator_selection_weights_are_always_valid():
    ops = [Operator("a"), Operator("b"), Operator("c")]
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), n_parents=1, operators=ops, log=False)
    ops = opt.operators
    ops[0].weight, ops[1].weight, ops[2].weight = -1.0, -0.2, -3.0
    w = opt._operator_selection_weights()
    assert min(w) >= 0 and sum(w) > 0
    ops[1].weight = float("inf")  # untried operator (UCB)
    assert opt._operator_selection_weights() == [0.0, 1.0, 0.0]


def test_adaptive_mutation_prompt_is_sent():
    opt = LLaMEA(
        scalar_f, ScriptedLLM(GOOD), adaptive_mutation=True, n_parents=1, log=False
    )
    parent = make(1.0, code="\n".join(f"x{i} = {i}" for i in range(40)))
    opt.population = [parent]
    prompt = opt.construct_prompt([parent], opt.operators[0])[0]["content"]
    assert "you can only change" in prompt


def test_role_prompt_is_separated_from_task_prompt():
    opt = LLaMEA(
        scalar_f, ScriptedLLM(GOOD), role_prompt="ROLE.", task_prompt="TASK", log=False
    )
    opt.population = [make(1.0)]
    content = opt.construct_prompt([make(1.0)], opt.operators[0])[0]["content"]
    assert content.startswith("ROLE.\nTASK")


def test_fitness_sharing_does_not_compound_over_generations():
    opt = LLaMEA(
        scalar_f, ScriptedLLM(GOOD), niching="sharing", niche_radius=1.0, log=False
    )
    s1, s2 = make(10.0), make(20.0)
    for _ in range(3):  # survivors are niched again every generation
        opt.apply_niching([s1, s2])
    assert s1.fitness == pytest.approx(5.0)
    assert s2.fitness == pytest.approx(10.0)


def test_reevaluated_fitness_becomes_new_raw_fitness():
    opt = LLaMEA(
        scalar_f, ScriptedLLM(GOOD), niching="sharing", niche_radius=1.0, log=False
    )
    s1, s2 = make(10.0), make(20.0)
    opt.apply_niching([s1, s2])
    s1.fitness = 30.0  # e.g. re-evaluated by evaluate_population
    opt.apply_niching([s1, s2])
    assert s1.get_metadata("raw_fitness") == 30.0


def test_best_so_far_reports_raw_fitness_with_niching():
    opt = LLaMEA(
        scalar_f, ScriptedLLM(GOOD), niching="sharing", niche_radius=1.0, log=False
    )
    s1, s2 = make(10.0), make(20.0)
    opt.population = opt.apply_niching([s1, s2])
    opt.update_best()
    assert opt.best_so_far.fitness == pytest.approx(20.0)


def test_parallel_errors_are_caught_and_budget_still_counts():
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), n_parents=2, log=False)

    def boom():
        raise TimeoutError("worker timed out")

    assert opt._run_parallel([delayed(boom)()]) == []
    placeholders = opt._timed_out_individuals(2)
    assert len(placeholders) == 2
    assert all(p.fitness == opt.worst_value for p in placeholders)
    assert "TimeoutError" in placeholders[0].error


def test_multi_objective_run_does_not_crash():
    def mo_f(solution, logger=None):
        solution.set_scores(Fitness({"a": 1.0, "b": 2.0}), "ok")
        return solution

    opt = LLaMEA(
        mo_f,
        ScriptedLLM(GOOD),
        n_parents=2,
        n_offspring=2,
        multi_objective=True,
        multi_objective_keys=["a", "b"],
        budget=6,
        max_workers=1,
        log=False,
    )
    front = opt.run()
    assert front and all(isinstance(s.fitness, Fitness) for s in front)


def test_feature_guidance_rejects_multi_objective():
    pytest.importorskip("xgboost")
    with pytest.raises(ValueError):
        LLaMEA(
            scalar_f,
            ScriptedLLM(GOOD),
            multi_objective=True,
            multi_objective_keys=["a"],
            feature_guided_mutation=True,
            log=False,
        )


def test_log_dir_name_sanitises_colon_and_slash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), experiment_name="x", log=True)
    assert ":" not in opt.logger.dirname
    assert "scripted_model_1" in opt.logger.dirname


def test_blade_hint_is_shown_once_and_can_be_disabled(capsys, monkeypatch):
    import llamea.llamea as core

    monkeypatch.setattr(core, "_BLADE_HINT_SHOWN", False)
    monkeypatch.setenv("LLAMEA_NO_BLADE_HINT", "1")
    core._show_blade_hint(scalar_f)
    assert "BLADE" not in capsys.readouterr().out
    monkeypatch.delenv("LLAMEA_NO_BLADE_HINT")
    core._show_blade_hint(scalar_f)
    core._show_blade_hint(scalar_f)
    assert capsys.readouterr().out.count("Tip:") == 1


# ---------------------------------------------------------------- LLM parsing


def test_name_is_the_class_not_a_word_containing_def():
    llm = Dummy_LLM()
    code = "# Undefined behaviour: none\ndef helper():\n    pass\n\nclass Algo(Base, Other):\n    pass"
    assert llm.extract_algorithm_name(code) == "Algo"
    assert llm.extract_algorithm_name("def only():\n    pass") == "only"
    with pytest.raises(NoCodeException):
        llm.extract_algorithm_name("x = 1")


def test_code_extraction_skips_a_leading_bash_block():
    llm = Dummy_LLM()
    message = (
        "Install:\n```bash\npip install x\n```\nNow the code:\n"
        "```python\nclass A:\n    pass\n```"
    )
    assert llm.extract_algorithm_code(message) == "class A:\n    pass"
    message = "```py\nclass B:\n    pass\n```\nThis uses x."
    assert llm.extract_algorithm_code(message) == "class B:\n    pass"


def test_empty_description_does_not_capture_next_line():
    llm = Dummy_LLM()
    assert llm.extract_algorithm_description("# Description:\n```python") == ""


def test_diff_mode_without_matching_block_raises():
    llm = ScriptedLLM(
        "```\n<<<<<<< SEARCH\nnot in code\n=======\nx = 2\n>>>>>>> REPLACE\n```"
    )
    with pytest.raises(ValueError):
        llm.sample_solution(
            [{"role": "user", "content": "go"}],
            base_code="class A:\n    x = 1",
            diff_mode=True,
        )


def test_diff_mode_applies_blocks_from_separate_fences():
    llm = ScriptedLLM(
        "```\n<<<<<<< SEARCH\n    x = 1\n=======\n    x = 2\n>>>>>>> REPLACE\n```\n"
        "and\n```\n<<<<<<< SEARCH\n    y = 1\n=======\n    y = 3\n>>>>>>> REPLACE\n```"
    )
    sol = llm.sample_solution(
        [{"role": "user", "content": "go"}],
        base_code="class A:\n    x = 1\n    y = 1",
        diff_mode=True,
    )
    assert sol.code == "class A:\n    x = 2\n    y = 3"


# ---------------------------------------------------------------- Solution


def test_configspace_survives_pickle_and_deepcopy():
    cs_mod = pytest.importorskip("ConfigSpace")
    sol = Solution(code="x", configspace=cs_mod.ConfigurationSpace({"a": (0.0, 1.0)}))
    for clone in (pickle.loads(pickle.dumps(sol)), copy.deepcopy(sol)):
        assert isinstance(clone.configspace, cs_mod.ConfigurationSpace)


def test_error_line_points_into_candidate_code():
    code = "import numpy as np\nx = 1\nnp.linalg.inv(np.zeros((2, 2)))"
    sol = Solution(code=code)
    try:
        exec(code, {})
    except Exception as e:
        sol.set_scores(0.0, error=e)
    assert "On line 3" in sol.error


def test_solutions_do_not_share_parent_id_lists():
    assert Solution().parent_ids is not Solution().parent_ids


# ---------------------------------------------------------------- namespace


def test_prepare_namespace_dotted_relative_and_versions():
    ns, issue = prepare_namespace(
        "import numpy.linalg\nfrom . import x\nimport typing", ["numpy==2.0"]
    )
    assert hasattr(ns["numpy"], "linalg")
    assert ns["typing"].__name__ == "typing"
    assert ".x" in issue


# ---------------------------------------------------------------- coverage of
# the remaining new branches (edge cases of the fixes above)


def test_prepare_namespace_star_and_submodule_imports():
    ns, issue = prepare_namespace("from math import *\nfrom xml import dom", ["xml"])
    assert "sqrt" in ns and issue is None
    assert ns["dom"].__name__ == "xml.dom"


def test_old_pickles_with_serialised_configspace_are_restored():
    cs_mod = pytest.importorskip("ConfigSpace")
    cs = cs_mod.ConfigurationSpace({"a": (0.0, 1.0)})
    sol = Solution(code="x")
    state = sol.__dict__.copy()
    state["configspace"] = cs.to_serialized_dict()  # what < 1.6.0 pickled
    state.pop("operator_id")
    restored = Solution.__new__(Solution)
    restored.__setstate__(state)
    assert isinstance(restored.configspace, cs_mod.ConfigurationSpace)
    assert restored.operator_id is None


def test_operator_equality_with_other_types():
    op = Operator("a")
    assert op != None  # noqa: E711
    assert op != "a"
    assert op == op


def test_custom_code_pattern_and_fallbacks():
    llm = Dummy_LLM(code_pattern=r"<code>(.*?)</code>")
    assert llm.extract_algorithm_code("x <code>class A: pass</code> y") == "class A: pass"
    with pytest.raises(NoCodeException):
        llm.extract_algorithm_code("no code")
    # only a non-Python block: use it rather than fail
    assert Dummy_LLM().extract_algorithm_code("```text\nclass T: pass\n```") == (
        "class T: pass"
    )


def test_diff_mode_without_markers_uses_the_full_code():
    llm = ScriptedLLM("```python\nclass B:\n    pass\n```")
    sol = llm.sample_solution(
        [{"role": "user", "content": "go"}], base_code="class A:\n    pass", diff_mode=True
    )
    assert sol.code == "class B:\n    pass" and sol.name == "B"


def test_extract_configspace():
    pytest.importorskip("ConfigSpace")
    message = "Space:\n```python\n{'x': (0.0, 1.0)}\n```"
    cs = Dummy_LLM().extract_configspace(message)
    assert cs is not None and "x" in list(cs.keys())


def _openai_error(cls, status, headers=None):
    import httpx

    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("error", response=response, body=None)


def _fake_response(content):
    from types import SimpleNamespace

    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_retries_transient_errors_only(monkeypatch):
    openai = pytest.importorskip("openai")
    from llamea.llm import OpenAI_LLM

    llm = OpenAI_LLM("key", "model", base_url="http://localhost:1/v1")
    assert llm._client_kwargs["base_url"] == "http://localhost:1/v1"
    sleeps = []
    monkeypatch.setattr("llamea.llm.time.sleep", sleeps.append)

    calls = iter(
        [
            _openai_error(openai.RateLimitError, 429, {"Retry-After": "1.5"}),
            _openai_error(openai.RateLimitError, 429, {"Retry-After": "Wed, 1 Jan"}),
            _fake_response(None),  # refusal: no content
        ]
    )

    def create(**kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    assert llm.query([{"role": "user", "content": "hi"}], default_delay=7) == ""
    assert sleeps == [1.5, 14]  # Retry-After, then default_delay * attempt

    def bad_request(**kwargs):
        raise _openai_error(openai.BadRequestError, 400)

    monkeypatch.setattr(llm.client.chat.completions, "create", bad_request)
    sleeps.clear()
    with pytest.raises(openai.BadRequestError):
        llm.query([{"role": "user", "content": "hi"}])
    assert sleeps == []  # not retried


def test_openai_temperature_none_is_omitted(monkeypatch):
    pytest.importorskip("openai")
    from llamea.llm import OpenAI_LLM

    llm = OpenAI_LLM("key", "model", temperature=None)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    assert llm.query([{"role": "user", "content": "hi"}]) == "ok"
    assert "temperature" not in seen


def test_gemini_retryable_errors():
    from llamea.llm import _gemini_retryable

    class E(Exception):
        def __init__(self, code):
            self.code = code

    assert _gemini_retryable(E(429)) and _gemini_retryable(E(503))
    assert _gemini_retryable(ConnectionError())
    assert not _gemini_retryable(E(400)) and not _gemini_retryable(E(404))


def test_log_solution_accepts_multi_objective_fitness(tmp_path, monkeypatch):
    import json

    import numpy as np

    from llamea.loggers import ExperimentLogger

    monkeypatch.chdir(tmp_path)
    logger = ExperimentLogger("mo")
    sol = Solution(code="x")
    sol.set_scores(Fitness({"a": 1.0, "b": np.float32(2.0)}))
    logger.log_solution(sol, np.array([1.0, 2.0]))
    with open(f"{logger.dirname}/solutions.jsonl") as fh:
        row = json.loads(fh.readline())
    assert row["fitness"] == {"a": 1.0, "b": 2.0} and row["output"] == [1.0, 2.0]


def test_restore_population_from_log_handles_odd_rows(tmp_path):
    import jsonlines

    rows = [
        {"code": "", "fitness": 1.0},  # no code: skipped
        {"code": "class A: pass", "name": "A", "fitness": None},
        {"code": "class B: pass", "name": "B", "fitness": 2.0, "feedback": "f"},
    ]
    with jsonlines.open(tmp_path / "log.jsonl", "w") as writer:
        writer.write_all(rows)
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), n_parents=2, log=False)
    opt.get_population_from(str(tmp_path))
    assert [s.name for s in opt.population] == ["A", "B"]
    assert opt.population[0].fitness == opt.worst_value

    mo_rows = [{"code": "class C: pass", "name": "C", "fitness": {"a": 1.0}}]
    with jsonlines.open(tmp_path / "log.jsonl", "w") as writer:
        writer.write_all(mo_rows)
    mo = LLaMEA(
        scalar_f,
        ScriptedLLM(GOOD),
        n_parents=1,
        multi_objective=True,
        multi_objective_keys=["a"],
        log=False,
    )
    mo.get_population_from(str(tmp_path))
    assert isinstance(mo.population[0].fitness, Fitness)


def test_clearing_off_interval_restores_raw_fitness():
    opt = LLaMEA(
        scalar_f,
        ScriptedLLM(GOOD),
        niching="clearing",
        niche_radius=1.0,
        clearing_interval=2,
        log=False,
    )
    best, worse = make(10.0), make(5.0)
    opt.generation = 2
    opt.apply_niching([best, worse])
    assert worse.fitness == opt.worst_value
    opt.generation = 3  # no clearing this generation
    opt.apply_niching([best, worse])
    assert worse.fitness == 5.0


def test_operator_weight_edge_cases():
    ops = [Operator("a"), Operator("b")]
    opt = LLaMEA(scalar_f, ScriptedLLM(GOOD), n_parents=1, operators=ops, log=False)
    a, b = opt.operators
    a.weight, b.weight = None, float("nan")
    assert opt._operator_selection_weights() == [1.0, 1.0]
    a.weight, b.weight = 0.0, 0.0
    assert opt._operator_selection_weights() == [1.0, 1.0]
    assert opt._run_parallel([]) == []

    class Boom(DefaultWeightUpdater):
        def update(self, id, reward):
            raise RuntimeError("boom")

    opt.operator_weight_updater = Boom([a.id, b.id])
    parent = make(1.0)
    opt.population = [parent]
    unknown_op = make(2.0)
    unknown_op.operator_id = "not-an-operator"
    orphan = make(2.0)
    orphan.operator_id, orphan.parent_ids = a.id, ["missing"]
    failing = make(2.0)
    failing.operator_id, failing.parent_ids = a.id, [parent.id]
    opt._update_operator_weights([unknown_op, orphan, failing])  # must not raise


def test_same_treats_nans_as_equal():
    from llamea.llamea import _same

    assert _same(float("nan"), float("nan"))
    assert not _same(float("nan"), 1.0)
    assert _same(1.0, 1.0)


def test_blade_hint_is_not_shown_inside_blade(capsys, monkeypatch):
    import llamea.llamea as core

    monkeypatch.setattr(core, "_BLADE_HINT_SHOWN", False)
    monkeypatch.delenv("LLAMEA_NO_BLADE_HINT", raising=False)
    FakeProblem = type("Problem", (), {"__module__": "iohblade.problem"})
    core._show_blade_hint(FakeProblem())
    assert capsys.readouterr().out == ""


def test_ucb_weights_start_at_exploration_score_and_refresh_for_all():
    ops = [Operator("a", weight=0.5), Operator("b", weight=0.5)]
    updater = DiscountedUCBState([o.id for o in ops])
    opt = LLaMEA(
        scalar_f,
        ScriptedLLM(GOOD),
        n_parents=1,
        operators=ops,
        operator_weight_updater=updater,
        log=False,
    )
    a, b = opt.operators
    assert a.weight == b.weight == float("inf")  # untried: explored first
    parent = make(1.0)
    opt.population = [parent]
    child = make(2.0)
    child.operator_id, child.parent_ids = a.id, [parent.id]
    opt._update_operator_weights([child])
    assert a.weight == pytest.approx(updater.score(a.id))
    assert b.weight == float("inf")  # still untried
    assert opt._operator_selection_weights() == [0.0, 1.0]


@pytest.mark.parametrize(
    "n_parents, n_offspring, budget", [(5, 5, 8), (5, 5, 3), (2, 4, 7)]
)
def test_budget_caps_the_number_of_evaluated_candidates(n_parents, n_offspring, budget):
    evaluated = []

    def counting_f(solution, logger=None):
        evaluated.append(solution)
        return scalar_f(solution)

    LLaMEA(
        counting_f,
        ScriptedLLM(GOOD),
        n_parents=n_parents,
        n_offspring=n_offspring,
        budget=budget,
        max_workers=1,
        log=False,
    ).run()
    assert len(evaluated) == budget
