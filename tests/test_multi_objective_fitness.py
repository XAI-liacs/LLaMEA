import math
from llamea.multi_objective_fitness import Fitness

def test_fitness_instantiates_properly():
    a = Fitness()
    assert a._fitness == {}

    fitness_dict: dict[str, float] = {
        'Distance' : 10,
        'Fuel': 18
    }
    b = Fitness(value=fitness_dict)
    assert b._fitness == fitness_dict

    #Value semantics followed test.
    fitness_dict['Distance'] = 100

    assert b._fitness != fitness_dict


def test_fitness_subscripting_works_properly():
    a = Fitness({
        'Distance': 100,
        'Fuel': 137
    })

    assert a['Distance'] == 100
    assert a['Fuel'] == 137

    # Return NAN, for un-available keys.
    assert math.isnan(a['Tyre Wear'])

def test_fitness_set_value_works_properly():
    a = Fitness()

    a['Distance'] = 1919
    a['Fuel'] = 2009

    assert a._fitness['Distance'] == 1919
    assert a._fitness['Fuel'] == 2009

o = Fitness({'Distance': 10, 'Fuel': 10})
q1 = Fitness({'Distance': 12, 'Fuel': 12})
q2 = Fitness({'Distance': 8, 'Fuel': 12})
q3 = Fitness({'Distance': 8, 'Fuel': 8})
q4 = Fitness({'Distance': 12, 'Fuel': 8})

def test_fitness_lt_comparison_returns_q1_fitness_true():
    assert (o < q1) == True
    assert (o < q2) == False
    assert (o < q3) == False
    assert (o < q4) == False

def test_fitness_gt_comparison_returns_q3_fitness_true():
    assert (o > q1) == False
    assert (o > q2) == False
    assert (o > q3) == True
    assert (o > q4) == False

def test_fitness_eq_comparison_returns_q2_q4_fitness_true():
    assert (o == q1) == False
    assert (o == q2) == True
    assert (o == q3) == False
    assert (o == q4) == True

def test_fitness_le_comparison_returns_q1_q2_s4_fitness_true():
    assert (o <= q1) == True
    assert (o <= q2) == True
    assert (o <= q3) == False
    assert (o <= q4) == True

def test_fitness_ge_comparison_returns_q2_q3_s4_fitness_true():
    assert (o >= q1) == False
    assert (o >= q2) == True
    assert (o >= q3) == True
    assert (o >= q4) == True