import math
from abc import ABC, abstractmethod


class WeightUpdater(ABC):
    @abstractmethod
    def __init__(self, operator_ids: list[str], **kwargs):
        """
        Instantiation of Weight Updater:

        ## Params:
            * `operator_ids: list[str]`: A list of `Operator.id` which are supposed to be tracked.
            * `**kwargs: dict[str, Any]`: Additional arguements for defining algorithm behaviour.
        """
        pass

    @abstractmethod
    def update(self, id: str, reward: float) -> float:
        """
        Weight update logic:

        ## Params:
        * `id: str`: ID of the instantiated operator in `__init__` for which the weight needs to be updated.
        * `reward: float`: A rewards value for the operator, often based on the offspring's performance compared to parent.

        ## Returns:
        * `weight: float`: The new weight of the operator.
        """
        pass


class DefaultWeightUpdater(WeightUpdater):
    """Always sets the weight of the operator = 1, mimicking random operator selector."""

    def __init__(self, operator_ids: list[str]):
        """
        Instantiation of Weight Updater:

        ## Params:
            * `operator_ids: list[str]`: A list of `Operator.id` which are supposed to be tracked.
        """
        self.weights: dict[str, float] = {}
        for operator_id in operator_ids:
            self.weights[operator_id] = 1.0

    def update(self, id: str, reward: float) -> float:
        """
        Updates the operator weight: Default behaviour is defined to always return weight = 1.

        ## Params:
            * `operator_ids: list[str]`: A list of `Operator.id` which are supposed to be tracked.
            * `**kwargs: dict[str, Any]`: Additional arguements for defining algorithm behaviour.
        """
        return self.weights[id]

    def score(self, id: str) -> float:
        return self.weights[id]


class DiscountedUCBState(WeightUpdater):
    """Returns discounted UBC scores, mimicking MCTS weight updates."""

    def __init__(self, operator_ids: list[str], gamma=0.95, c=1.0):
        """
        Instantiation of Discounted UBC Weight Updater:

        ## Params:
            * `operator_ids: list[str]`: A list of `Operator.id` which are supposed to be tracked.
            * `gamma: float(0.95)`: Reward discount factor, higher the gamma, slower the discount will be applied over iterations.
            * `c: float(1.0): Weightage of the bonus calculation for score calculation.
        """
        self.gamma = gamma
        self.c = c

        self.sum_rewards = {op: 0.0 for op in operator_ids}
        self.sum_weights = {op: 0.0 for op in operator_ids}
        self.total_weight = 0.0

    def score(self, op_id) -> float:
        """
        Returns the score/weight of the provided operator id.

        ## Params:
        * `op_id: str`: Operator id for which the wight has to be returned (without updating.)

        ## Returns:
        * `weight: float`: Weight of the provided operator id.
        """
        w = self.sum_weights[op_id]
        if w == 0.0:
            return float("inf")

        mean = self.sum_rewards[op_id] / w
        bonus = self.c * math.sqrt(math.log(max(1.0, self.total_weight)) / w)
        return mean + bonus

    def update(self, operator_id: str, reward: float) -> float:
        """
        Returns the score/weight of the provided operator id after updating their scores.

        ## Params:
        * `op_id: str`: Operator id for which the wight has to be returned (after updating.)
        * `reward: float`: Rewards for the operator based on how much better the offspring's fitness is compared to the parent(s).

        ## Returns:
        * `weight: float`: Weight of the provided operator id.
        """
        for k in self.sum_rewards:
            self.sum_rewards[k] *= self.gamma
            self.sum_weights[k] *= self.gamma

        self.total_weight *= self.gamma

        # add new reward
        self.sum_rewards[operator_id] += reward
        self.sum_weights[operator_id] += 1.0
        self.total_weight += 1.0
        return self.score(operator_id)

    def snapshot(self):
        """
        DEBUG FUNCTION:
            * Prints the current operator weights and diagonstics.
        """
        rows = []
        for op in self.sum_rewards:
            w = self.sum_weights[op]
            mean = self.sum_rewards[op] / w if w > 0 else 0.0
            score = self.score(op)
            rows.append((op, score, mean, w))

        rows.sort(key=lambda x: x[1], reverse=True)

        print("\n[UCB] operator scores:")
        for op, s, m, w in rows:
            print(f"  {op:12s} score={s:6.3f} mean={m:6.3f} n={w:5.1f}")
