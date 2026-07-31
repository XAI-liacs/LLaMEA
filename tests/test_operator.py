from llamea import Fitness
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
    operator = Operator(prompt="Hi there")

    assert operator.name == 'mutation'
    assert operator.prompt == "Hi there"
    assert operator.number_of_parents == 1
    assert operator.weight == 0.5



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
        _ = LLaMEA(f, llm, 5, 5, operators=operators, log=False)

    operators = [
        Operator(
            'crossover',
            'Combine the strengths of these algorithms, and generate a completely new solution.',
            -1.0,
            2
        )
    ]
    with pytest.raises(AssertionError):
        _ = LLaMEA(f, llm, 5, 5, operators=operators, log=False)

def test_prompt():

    llm = Dummy_LLM()

    operator = Operator(
        name='crossover',
        prompt = 'Pick the best features of these individuals, and generate a new algorithm that combines these strengths.',
        number_of_parents=3,
        weight=1.0
    )

    solution1 = Solution(
'''
def adder(a, b):
    return a + b
''',
    'adder',
    'A calculator module for adding.'
    )
    solution1.set_scores(1, 'Adder works as expected.')
    solution2 = Solution(
'''
def substractor(a, b):
    return a - b
''',
    'substractor',
    'A calculator module for substracting.'
    )
    solution2.set_scores(1, 'Substractor works as expected.')

    solution3 = Solution(
'''
def summation(upto):
    return upto * (upto + 1) / 2
''',
    'summation',
    'A calculator module for summation.'
    )
    solution3.set_scores(0, 'Summation returns double nom.', ValueError("A summation always must return int."))

    llamea = LLaMEA(
        f,
        llm,
        operators=[operator],
        log=False
    )

    message = llamea.construct_prompt([solution1, solution2, solution3], operator)[0]['content']

    for solution in [solution1, solution2, solution3]:
        assert solution.code in message
        assert solution.description in message
        assert solution.feedback in message
        if solution.error:
            assert solution.error in message

    assert operator.prompt or '' in message


def test_operator_picker_works_correctly(monkeypatch):
    llm = Dummy_LLM()

    operator1 = Operator(
        'mutation',
        'Update the mentioned individual using small but useful changes.',
        0.2,
        1,
    )
    operator2 = Operator(
        'mutation',
        'Update the mentioned individual big changes.',
        0.4,
        1,
    )
    operator3 = Operator(
        'crossover',
        'Combine the strengths of the mentioned individuals, and write an algorithm that reflects those choices.',
        0.6,
        4,
    )

    llamea = LLaMEA(
        f,
        llm,
        operators=[operator1, operator2, operator3],
        log=False
    )

    captured = {}

    def fake_choices(population, weights=None, *, cum_weights=None, k=1):
        captured["population"] = population
        captured["weights"] = weights
        captured["cum_weights"] = cum_weights
        captured["k"] = k
        return [population[1]]  # Always pick index 1

    monkeypatch.setattr(random, "choices", fake_choices)

    captured_2 = {}
    def fake_evolve_solution(individual, operator):
        captured_2['individual'] = individual
        captured_2['operator'] = operator
        return Solution('''func Summation(upto n: Int) -> Int
{
    return n * (n + 1) / 2
}''', 'Summation Oeprtor', 'An O(1) solution for finding sum 1, to N^+')

    # Call the code under test
    monkeypatch.setattr(llamea, '_evolve_solution', fake_evolve_solution)
    for _ in range(5):
        llamea.population.append(Solution())

    selected = llamea.evolve_solution(random.choice(llamea.population))

    assert "func Summation(upto n: Int) -> Int" in selected.code
    assert captured["population"] == [operator1, operator2, operator3]
    assert captured["weights"] is not None  # or assert the exact weights
    assert captured["cum_weights"] is None
    assert captured["k"] == 1
    assert captured_2['individual'] in llamea.population
    assert captured_2['operator'] is operator2

def test_post_init_script_triggered():
    o1 = Operator('Hi there', 'greetings', 1.0, 2)
    getattr(o1, 'id', None) is not None

def test_operator_rewards_ignore_invalid_offspring():

    o1 = Operator('Refine the program, by optimising the hyper-parameters', 'mutation', 0.5, 1)
    o2 = Operator('Refine the program, by optimising the methods', 'mutation', 0.5, 1)
    o3 = Operator('Combine the strengths of the provided algorithm.', 'mutation', 0.5, 2)

    llm = Dummy_LLM()

    llamea = LLaMEA(f, llm, operators=[o1, o2, o3], log=False)

    offspring = Solution()

    parent1 = Solution()
    parent1.set_scores(0.85, 'Scored 0.85, best solution 1.0')
    parent2 = Solution()
    parent2.set_scores(0.76, 'Scored 0.76, best solution 1.0')
    parent3 = Solution()
    parent3.set_scores(0.22, 'Scored 0.22, best solution 1.0')

    score = llamea._operator_rewards([parent1], offspring)
    assert score == -1

    score = llamea._operator_rewards([parent1, parent2], offspring)
    assert score == -2

def test_operator_rewards_scalar_fitness_properly():

    o1 = Operator('Refine the program, by optimising the hyper-parameters', 'mutation', 0.5, 1)
    o2 = Operator('Refine the program, by optimising the methods', 'mutation', 0.5, 1)
    o3 = Operator('Combine the strengths of the provided algorithm.', 'mutation', 0.5, 3)

    llm = Dummy_LLM()

    llamea = LLaMEA(f, llm, operators=[o1, o2, o3], log=False)

    offspring = Solution()
    offspring.set_scores(0.55, 'Scored 0.55, best solution 1.0')

    parent1 = Solution()
    parent1.set_scores(float('nan'), 'Compile time error.')
    parent2 = Solution()
    parent2.set_scores(0.76, 'Scored 0.76, best solution 1.0')
    parent3 = Solution()
    parent3.set_scores(0.22, 'Scored 0.22, best solution 1.0')

    score = llamea._operator_rewards([parent1], offspring)
    assert score == 1

    score = llamea._operator_rewards([parent1, parent2, parent3], offspring)
    assert score == 1

def test_operator_rewards_multi_objective_fitness_fitness_properly():

    o1 = Operator('Refine the program, by optimising the hyper-parameters', 'mutation', 0.5, 1)
    o2 = Operator('Refine the program, by optimising the methods', 'mutation', 0.5, 1)
    o3 = Operator('Combine the strengths of the provided algorithm.', 'mutation', 0.5, 3)

    llm = Dummy_LLM()

    llamea = LLaMEA(f, llm, operators=[o1, o2, o3], multi_objective=True, multi_objective_keys=['Distance', 'Fuel'], log=False, minimization=True)

    offspring = Solution()
    fitness = Fitness({'Distance': 983, 'Fuel': 1121})
    offspring.set_scores(fitness, f'Scored {fitness}')

    parent1 = Solution()
    parent1.set_scores(float('nan'), 'Compile time error.')
    parent2 = Solution()
    fitness2 = Fitness({'Distance': 911, 'Fuel': 1101})
    parent2.set_scores(fitness2, f'Scored {fitness2}')
    parent3 = Solution()
    fitness3 = Fitness({'Distance': 2011, 'Fuel': 3101})
    parent3.set_scores(fitness3, f'Scored {fitness3}')

    score = llamea._operator_rewards([parent1], offspring)
    assert score == 1

    score = llamea._operator_rewards([parent1, parent2, parent3], offspring)
    assert score == 1

def test_default_weight_updater():

    o1 = Operator('Refine the program, by optimising the hyper-parameters', 'mutation', 0.5, 1)
    o2 = Operator('Refine the program, by optimising the methods', 'mutation', 0.5, 1)
    o3 = Operator('Combine the strengths of the provided algorithm.', 'mutation', 0.5, 3)

    llm = Dummy_LLM()

    llamea = LLaMEA(f, llm, operators=[o1, o2, o3], log=False)

    offspring = Solution()
    offspring.set_scores(0.55, 'Scored 0.55, best solution 1.0')

    parent1 = Solution()
    parent1.set_scores(float('nan'), 'Compile time error.')
    parent2 = Solution()
    parent2.set_scores(0.76, 'Scored 0.76, best solution 1.0')
    parent3 = Solution()
    parent3.set_scores(0.22, 'Scored 0.22, best solution 1.0')

    o1weight = o1.weight
    llamea.update_weight(llamea.operators[0], [parent1], offspring)
    assert o1weight < llamea.operators[0].weight

    o3weight = o3.weight
    llamea.update_weight(llamea.operators[2], [parent1, parent2, parent3], offspring)
    assert o3weight < llamea.operators[2].weight

def test_llamea_pickles_when_member_from_other_class_is_provided_as_function():
    import pickle
    from misc.ubc_weight_update import DiscountedUCBState

    o1 = Operator('Refine the program, by optimising the hyper-parameters', 'mutation', 0.5, 1)
    o2 = Operator('Refine the program, by optimising the methods', 'mutation', 0.5, 1)
    o3 = Operator('Combine the strengths of the provided algorithm.', 'mutation', 0.5, 3)

    ucb = DiscountedUCBState([operator.id for operator in [o1, o2, o3]])

    llm = Dummy_LLM()

    llamea = LLaMEA(f, llm, operators=[o1, o2, o3], log=False, operator_weight_update_function=ucb.update)
    _ = pickle.dumps(llamea)
