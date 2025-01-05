import numpy as np

class DQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0

    def _quantum_superposition(self, pos1, pos2):
        return 0.5 * (pos1 + pos2) + np.random.normal(0, 0.1, self.dim)

    def _mutate(self, idx, func):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.positions[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.f * (b - c)
        mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
        return mutant

    def _crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.cr:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        scores = np.array([func(pos) for pos in self.positions])
        self.gbest_score = scores.min()
        self.gbest = self.positions[scores.argmin()]
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i, func)
                trial = self._crossover(self.positions[i], mutant)
                trial_score = func(trial)

                if trial_score < scores[i]:
                    scores[i] = trial_score
                    self.positions[i] = trial

                if trial_score < self.gbest_score:
                    self.gbest_score = trial_score
                    self.gbest = trial

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return self.gbest

            # Dynamic strategy adaptation
            if self.evaluations % (self.budget // 5) == 0:
                self.f = max(0.2, self.f * 0.9)
                self.cr = min(1.0, self.cr + 0.05)

        return self.gbest