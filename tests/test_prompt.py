import pytest
import random
from llamea import LLaMEA, Dummy_LLM, Operator, Solution

def f(solution: Solution):
    evaluation = random.random()
    solution.set_scores(
        evaluation,
        f'Got {evaluation}, best known 1.2.'
    )
    return solution

def test_operator_null_handling():
    operator = Operator(None, "Hi there", None, None)

    assert operator.name == 'mutation'
    assert operator.prompt == "Hi there"
    assert operator.number_of_parents == 1
    assert operator.weight == 1



def test_operator_asserts_proper_parent_count():

    llm = Dummy_LLM()
    operators = [
        Operator(
            'crossover',
            'Combine the strengths of these algorithms, and generate a completely new solution.',
            1.0,
            10
        )
    ]

    with pytest.raises(AssertionError):
        _ = LLaMEA(f, llm, 5, 5, operators=operators)

    operators = [
        Operator(
            'crossover',
            'Combine the strengths of these algorithms, and generate a completely new solution.',
            -1.0,
            2
        )
    ]
    with pytest.raises(AssertionError):
        _ = LLaMEA(f, llm, 5, 5, operators=operators)
    
def test_crossover_operator():
    