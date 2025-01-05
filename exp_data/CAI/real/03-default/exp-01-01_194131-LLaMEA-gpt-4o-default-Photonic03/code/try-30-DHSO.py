import numpy as np

class DHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, 6 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.alpha = 0.5 + np.random.rand(self.population_size, 1)
        self.evaluations = 0
        self.hierarchy_levels = max(3, dim // 4)
        self.local_gbests = [None] * self.hierarchy_levels
        self.local_gbest_scores = [float('inf')] * self.hierarchy_levels

    def gaussian_perturbation(self, scale=0.1):
        return np.random.normal(0, scale, self.dim)

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        level = idx % self.hierarchy_levels
        local_gbest = self.local_gbests[level] if self.local_gbests[level] is not None else self.gbest
        self.velocities[idx] = (self.alpha[idx] * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (local_gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]

        if np.random.rand() < 0.1:  # Intensification with Gaussian perturbation
            new_pos += self.gaussian_perturbation()

        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)

        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score

        if new_score < self.local_gbest_scores[level]:
            self.local_gbests[level] = new_pos
            self.local_gbest_scores[level] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def _adapt_hierarchy(self):
        if self.evaluations % (self.budget // 5) == 0:
            new_hierarchy_levels = min(self.hierarchy_levels + 1, 5 * self.dim)
            if new_hierarchy_levels > self.hierarchy_levels:
                self.local_gbests.extend([None] * (new_hierarchy_levels - self.hierarchy_levels))
                self.local_gbest_scores.extend([float('inf')] * (new_hierarchy_levels - self.hierarchy_levels))
                self.hierarchy_levels = new_hierarchy_levels

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
            self._adapt_hierarchy()

        return self.gbest