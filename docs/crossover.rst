Crossover
~~~~~~~~~~

Crossover operators define how new individuals are generated from one or more existing individuals. They are configured using the Operators class and can be supplied to LLaMEA through the operators parameter.

Operators

An operator is defined as follows:

.. code-block:: python

    from dataclasses import dataclass, field
    import uuid

    @dataclass
    class Operators:
        prompt: str
        name: str = "mutation"
        weight: float = field(default=0.5, compare=False)
        number_of_parents: int = 1

        def __post_init__(self):
            self.id = uuid.uuid4().hex
            self.rewards = 0.0


Parameters

``prompt`` (``str``)
    Instructions describing the operation to perform. The prompt is appended to the end of the ``LLaMEA.task_prompt`` when the operator is selected.
``name`` (``str``, default: ``"mutation"``)
    The name of the operator. Use this to identify or distinguish different mutation and crossover strategies.
``weight`` (``float``, default: ``0.5``)
    Determines the relative probability that the operator will be selected. Operators with larger weights are more likely to be selected than operators with smaller weights.
    The weight must be a non-negative real number.
``number_of_parents`` (``int``, default: ``1``)
    Specifies how many individuals from the parent population are used to construct the prompt for generating the next individual.
    Valid values are ``1 <= number_of_parents <= parent_population``
``id`` (``str``)
    A unique identifier automatically generated during initialization using ``uuid.uuid4().hex``. It is used by the ``WeightUpdater`` to associate rewards and weights with individual operators.
``rewards`` (``float``)
    Stores the reward accumulated by the operator. Rewards are provided by LLaMEA based on the performance of offspring relative to their parent(s).

When providing operators with `number_of_parents > 1`, one an also use LLaMEA's parameter:
``parent_selection: str``

For operators with number_of_parents > 1, use the parent_selection parameter to specify how parents are selected.

    Supported strategies are:
        * "random"(default) – randomly selects parents from the parent population.
        * "tournament" – selects parents using tournament-based selection.
        * "roulette" – selects parents according to their relative fitness.

For example:

.. code-block:: python

    operator = Operators(
        prompt="Combine useful properties from the selected parents.",
        name="crossover"",
        number_of_parents=2,
    )

Using operators with LLaMEA

Operators are provided to LLaMEA as a list:
    .. code-block:: python

        operators = [
            mutation_operator,
            crossover_operator,
        ]
        llamea = LLaMEA(
            ...,
            operators=operators,
            parent_selection='roulette',
            ...
        )

WeightUpdater
-------------
``WeightUpdater`` is an abstract base class responsible for dynamically updating operator weights based on their observed performance.
Implementations must provide the following two methods:

.. code-block:: python

   class WeightUpdater(ABC):
       def __init__(self, operator_ids: list[str]) -> object:
           ...
       def update(self, id: str, reward: float) -> float:
           ...

``__init__``

The initializer receives a list of operator IDs:

.. code-block:: python

    operator_ids: list[str]

These IDs are used to maintain state for each operator.

``update``

The ``update`` method receives:
    * ``id`` -- the ID of the operator whose weight should be updated.
    * ``reward`` -- the reward assigned to the operator by LLaMEA.

The reward reflects the performance of the generated offspring relative to its parent or parents.
The method returns the operator's updated weight.

DefaultWeightUpdater
--------------------
``DefaultWeightUpdater`` provides the default weight-update behaviour.
It does not adapt operator weights based on rewards. Whenever an operator is updated, its weight is set to `1.0`.
This results in equal relative selection weights for all operators using this updater.

DiscountedUCBState
------------------
``DiscountedUCBState`` is a ``WeightUpdater`` implementation based on a discounted Upper Confidence Bound (UCB) strategy, similar to the approach used in Monte Carlo Tree Search.
Unlike ``DefaultWeightUpdater``, it adapts operator weights according to their observed rewards while accounting for both exploitation and exploration.

Configuration

* ``gamma`` (``float``, default: ``0.95``)
    Reward discount factor. A higher value gives more importance to historical rewards, resulting in slower discounting of past observations.
* ``c`` (``float``, default: ``1.0``)
    Exploration coefficient controlling the contribution of the UCB exploration bonus. Higher values encourage greater exploration of less frequently selected operators.

For example:

.. code-block:: python

    weight_updater = DiscountedUCBState(
       operator_ids=[operator.id for operator in operators],
       gamma=0.95,
       c=1.0,
   )

Configuring weight updaters

* Any instance of a class derived from ``WeightUpdater`` can be passed to LLaMEA using the ``operator_weight_updater`` parameter:

.. code-block:: python

    llamea = LLaMEA(
       ...,
       operators=operators,
       operator_weight_updater=weight_updater,
   )

This allows for custom implementation of operator weight update strategy.
