import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.phi = np.log(2)
        self.beta = 0.3
        self.evaluations = 0
        self.dynamic_neighborhood_size = max(3, dim // 3)

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        neighbors = np.random.choice(self.population_size, self.dynamic_neighborhood_size, replace=False)
        local_gbest = min(neighbors, key=lambda n: self.pbest_scores[n])
        local_gbest_position = self.pbest[local_gbest]

        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (local_gbest_position - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]

        if np.random.rand() < self.beta:
            new_pos += self.levy_flight()

        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)

        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def _adapt_dynamic_parameters(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.dynamic_neighborhood_size = min(self.dynamic_neighborhood_size + 1, self.population_size // 2)
            self.beta = min(1.0, self.beta + 0.05)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.ub) * np.random.rand(self.population_size, self.dim)
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
            self._adapt_dynamic_parameters()

        return self.gbest