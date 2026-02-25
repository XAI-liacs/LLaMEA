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
    assert (o == q2) == False
    assert (o == q3) == False
    assert (o == q4) == False
    assert (o == o) == True

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

a1 = Fitness({'Distance': 10, 'Fuel': 12})
a2 = Fitness({'Distance': 12, 'Fuel': 10})
a3 = Fitness({'Distance': 8, 'Fuel': 10})
a4 = Fitness({'Distance': 10, 'Fuel': 8})

def test_fitness_lt_comparison_returns_negative_axis_fitness_false():
    assert (o < a1) == True
    assert (o < a2) == True
    assert (o < a3) == False
    assert (o < a4) == False

def test_fitness_gt_comparison_returns_positive_axis_fitness_true():
    assert (o > a1) == False
    assert (o > a2) == False
    assert (o > a3) == True
    assert (o > a4) == True

def test_fitness_eq_comparison_returns_all_axis_fitness_false():
    assert (o == a1) == False
    assert (o == a2) == False
    assert (o == a3) == False
    assert (o == a4) == False
    assert (o == o) == True

def test_fitness_le_comparison_returns_positive_axis_fitness_true():
    assert (o <= a1) == True
    assert (o <= a2) == True
    assert (o <= a3) == False
    assert (o <= a4) == False

def test_fitness_ge_comparison_returns_negative_axis_fitness_true():
    assert (o >= a1) == False
    assert (o >= a2) == False
    assert (o >= a3) == True
    assert (o >= a4) == True

def test_vectorisation():
    # Return empty on un-evaluated fitness.
    f1 = Fitness()
    assert f1.to_vector() == []

    # Releases only values as vector.
    f2 = Fitness({"Flexibility": 10, "Scratch Resistance": 2})
    assert f2.to_vector() == [10, 2]

    # Sorts key for consistenty.
    f3 = Fitness({'Fuel': 228, 'Distance': 110})
    assert f3.to_vector() == [110, 228]

def test_is_json_serialisable():
    import json
    data = json.dumps(o, default=Fitness.to_dict)
    data = json.loads(data, object_hook=Fitness.from_dict)
    assert isinstance(data, Fitness)