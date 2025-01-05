import numpy as np

class QEDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.alpha = 0.5
        self.delta = 0.2
        self.evaluations = 0
        self.local_search_pivot = np.random.rand(self.dim)

    def _quantum_exploration(self, pos):
        shift = np.random.normal(0, 1, self.dim) * (np.abs(pos - self.local_search_pivot) ** 2)
        return pos + shift

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        quantum_jumps = self._quantum_exploration(self.positions[idx])
        inertia = (1 - self.alpha) * self.velocities[idx] + self.alpha * (self.pbest[idx] - self.positions[idx])
        cooperation = r1 * (self.pbest[idx] - self.positions[idx]) + r2 * (self.gbest - self.positions[idx])
        self.velocities[idx] = inertia + cooperation
        new_pos = self.positions[idx] + self.velocities[idx]
        new_pos = np.clip(new_pos + quantum_jumps, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)
        
        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def _adapt_local_search_pivot(self):
        if self.evaluations % (self.budget // 5) == 0:
            best_idx = np.argmin(self.pbest_scores)
            self.local_search_pivot = self.pbest[best_idx] + self.delta * (np.random.rand(self.dim) - 0.5)

    def _adjust_alpha(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.alpha = max(0.1, self.alpha - 0.05)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.pbest[i] = self.positions[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_particle(i, func)
                if self.evaluations >= self.budget:
                    break
            self._adapt_local_search_pivot()
            self._adjust_alpha()

        return self.gbest