import numpy as np

class EnhancedQAPSO:
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
        self.num_swarms = max(2, dim // 5)
        self.local_gbests = [None] * self.num_swarms
        self.local_gbest_scores = [float('inf')] * self.num_swarms
        self.elite_fraction = 0.2  # Fraction of top individuals to focus on
        self.adaptive_neighborhood = 0.1 + 0.9 * (budget // 1000)  # Scales with budget

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        swarm_idx = idx % self.num_swarms
        local_gbest = self.local_gbests[swarm_idx] if self.local_gbests[swarm_idx] is not None else self.gbest
        new_velocity = self.phi * self.velocities[idx] + \
                       r1 * (self.pbest[idx] - self.positions[idx]) + \
                       r2 * (local_gbest - self.positions[idx])
        self.velocities[idx] = new_velocity
        new_pos = self.positions[idx] + new_velocity

        if np.random.rand() < self.beta:
            new_pos += self.levy_flight()

        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)

        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score

        if new_score < self.local_gbest_scores[swarm_idx]:
            self.local_gbests[swarm_idx] = new_pos
            self.local_gbest_scores[swarm_idx] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def _dynamic_neighborhood(self):
        neighborhood_size = int(self.population_size * self.adaptive_neighborhood)
        for i in range(self.population_size):
            neighborhood = np.random.choice(self.population_size, neighborhood_size, replace=False)
            for neighbor in neighborhood:
                if self.pbest_scores[neighbor] < self.pbest_scores[i]:
                    self.pbest[i] = self.pbest[neighbor]
                    self.pbest_scores[i] = self.pbest_scores[neighbor]
                    
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
            elite_indices = np.argsort(self.pbest_scores)[:int(self.elite_fraction * self.population_size)]
            for i in elite_indices:
                self._update_particle(i, func)
                if self.evaluations >= self.budget:
                    break
            self._dynamic_neighborhood()

        return self.gbest