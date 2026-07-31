import math

class DiscountedUCBState:
    def __init__(self, operator_ids, gamma=0.95, c=1.0):
        self.gamma = gamma
        self.c = c

        self.sum_rewards = {op: 0.0 for op in operator_ids}
        self.sum_weights = {op: 0.0 for op in operator_ids}
        self.total_weight = 0.0

    def score(self, op_id):
        w = self.sum_weights[op_id]
        if w == 0.0:
            return float("inf")

        mean = self.sum_rewards[op_id] / w
        bonus = self.c * math.sqrt(
            math.log(max(1.0, self.total_weight)) / w
        )
        return mean + bonus

    def update(self, operator, reward) -> float:
        # discount everything
        for k in self.sum_rewards:
            self.sum_rewards[k] *= self.gamma
            self.sum_weights[k] *= self.gamma

        self.total_weight *= self.gamma

        # add new reward
        self.sum_rewards[operator.id] += reward
        self.sum_weights[operator.id] += 1.0
        self.total_weight += 1.0
        return self.score(operator.id)

    def snapshot(self):
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
