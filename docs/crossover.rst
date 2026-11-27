Operators: mutation and crossover
=================================

An :class:`~llamea.operator.Operator` is a prompt that tells the LLM how to
turn one or more selected parents into a new algorithm. Every offspring is
created by one operator, drawn at random in proportion to the operator
weights.

.. code-block:: python

   from llamea import LLaMEA, Operator

   operators = [
       Operator(prompt="Refine the strategy of the selected solution to improve it.",
                name="mutation", weight=0.4),
       Operator(prompt="Combine the selected solutions into a better algorithm.",
                name="crossover", weight=0.6, number_of_parents=2),
   ]
   optimizer = LLaMEA(evaluate, llm, n_parents=4, n_offspring=8,
                      operators=operators, parent_selection="tournament")

Plain strings are accepted too: ``operators=["Refine ...", "Redesign ..."]``
creates single-parent mutation operators with weight 1.

Fields
------

``prompt`` (``str``)
    The instruction, appended after the selected parents in the prompt.
``name`` (``str``, default ``"mutation"``)
    A label shown in the logs (``solution.operator``).
``weight`` (``float``, default ``0.5``)
    Relative probability of being picked. Must be non-negative.
``number_of_parents`` (``int``, default ``1``)
    Parents shown to the LLM; at most ``n_parents``. Besides the parent that
    is being varied, ``number_of_parents - 1`` further parents are chosen
    with ``parent_selection``.

Parent selection
----------------

``parent_selection`` chooses the parents of each offspring:

* ``"random"`` (default): uniformly, with replacement;
* ``"tournament"``: best of ``tournament_size`` random candidates;
* ``"roulette"``: rank-based roulette wheel (robust to negative or infinite
  fitness).

Adaptive operator selection
---------------------------

By default the weights stay as configured. Pass an
``operator_weight_updater`` to adapt them during the run: after each
generation, every offspring earns its operator a **reward** of +1 for each
parent it beats, -1 for each parent that beats it and 0 otherwise (Pareto
dominance for several objectives; a failed offspring scores ``-number of
parents``). The updater turns the rewards into new weights.

.. code-block:: python

   from llamea import DiscountedUCBState

   updater = DiscountedUCBState([op.id for op in operators], gamma=0.95, c=1.0)
   optimizer = LLaMEA(evaluate, llm, operators=operators,
                      operator_weight_updater=updater)

:class:`~llamea.weight_updaters.DiscountedUCBState`
    Discounted upper confidence bound: the mean discounted reward plus an
    exploration bonus. ``gamma`` controls how quickly old rewards are
    forgotten, ``c`` the amount of exploration. Operators that were never
    tried are tried first; negative scores are shifted so that all weights
    remain valid.
:class:`~llamea.weight_updaters.DefaultWeightUpdater`
    Sets every weight to 1 after an update (uniform selection).

A custom strategy subclasses :class:`~llamea.weight_updaters.WeightUpdater`:

.. code-block:: python

   from llamea.weight_updaters import WeightUpdater

   class Greedy(WeightUpdater):
       def __init__(self, operator_ids):
           self.total = {i: 0.0 for i in operator_ids}

       def update(self, id, reward):
           self.total[id] += reward
           return max(self.total[id], 0.0) + 0.1

.. note::
   Before LLaMEA 1.6.0 the ``operator_weight_updater`` argument was accepted
   but never used, so operator weights did not adapt.
