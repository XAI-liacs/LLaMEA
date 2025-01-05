import numpy as np

class EQAPSO:
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
        self.chaotic_map = np.random.rand()

    def _chaotic_beta(self):
        # Chaotic map to adapt beta for improved exploration-exploitation
        self.chaotic_map = 4 * self.chaotic_map * (1 - self.chaotic_map)
        self.beta = 0.2 + 0.8 * self.chaotic_map

    def _diversity_preservation(self):
        # Preserve diversity by perturbing some positions
        if self.evaluations % (self.budget // 5) == 0:
            num_perturb = self.population_size // 5
            indices = np.random.choice(self.population_size, num_perturb, replace=False)
            for idx in indices:
                self.positions[idx] += 0.1 * (np.random.rand(self.dim) - 0.5)
                self.positions[idx] = np.clip(self.positions[idx], func.bounds.lb, func.bounds.ub)

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (self.gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]

        if np.random.rand() < self.beta:
            new_pos = (self.pbest[idx] + self.gbest) / 2 + np.abs(self.gbest - self.positions[idx]) * np.random.normal(size=self.dim)

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
            self._chaotic_beta()
            self._diversity_preservation()

        return self.gbest