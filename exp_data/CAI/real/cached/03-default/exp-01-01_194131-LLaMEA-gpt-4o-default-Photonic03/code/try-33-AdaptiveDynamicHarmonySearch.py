import numpy as np

class AdaptiveDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMS = max(10, 2 * dim)  # Harmony Memory Size
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3  # Pitch Adjustment Rate
        self.HM = np.random.rand(self.HMS, dim)  # Harmony Memory
        self.HM_scores = np.full(self.HMS, float('inf'))
        self.best_harmony = None
        self.best_score = float('inf')
        self.evaluations = 0

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _adaptive_PAR(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.PAR = min(0.5, self.PAR + 0.05)

    def _generate_harmony(self, func):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                idx = np.random.randint(0, self.HMS)
                new_harmony[i] = self.HM[idx, i]
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.uniform(-0.1, 0.1)
            else:
                new_harmony[i] = np.random.uniform(func.bounds.lb[i], func.bounds.ub[i])

        new_harmony += self.levy_flight()
        new_harmony = np.clip(new_harmony, func.bounds.lb, func.bounds.ub)

        return new_harmony

    def __call__(self, func):
        self.HM = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.HMS, self.dim)
        for i in range(self.HMS):
            score = func(self.HM[i])
            self.HM_scores[i] = score
            if score < self.best_score:
                self.best_harmony = self.HM[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_harmony

        while self.evaluations < self.budget:
            new_harmony = self._generate_harmony(func)
            new_score = func(new_harmony)
            self.evaluations += 1

            if new_score < self.best_score:
                self.best_harmony = new_harmony
                self.best_score = new_score

            worst_idx = np.argmax(self.HM_scores)
            if new_score < self.HM_scores[worst_idx]:
                self.HM[worst_idx] = new_harmony
                self.HM_scores[worst_idx] = new_score

            self._adaptive_PAR()

            if self.evaluations >= self.budget:
                break

        return self.best_harmony