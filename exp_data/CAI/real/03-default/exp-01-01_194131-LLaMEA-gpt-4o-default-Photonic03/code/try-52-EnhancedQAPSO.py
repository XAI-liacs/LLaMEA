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
        self.tol = 1e-5  # Stagnation tolerance

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / 3))

    def hybrid_local_search(self, position, func, max_iters=10):
        step_size = 0.1
        for _ in range(max_iters):
            new_pos = position + step_size * np.random.uniform(-1, 1, self.dim)
            new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
            new_score = func(new_pos)
            if new_score < func(position):
                position = new_pos
        return position

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        swarm_idx = idx % self.num_swarms
        local_gbest = self.local_gbests[swarm_idx] if self.local_gbests[swarm_idx] is not None else self.gbest
        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (local_gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]

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

    def _adapt_population(self):
        if self.evaluations % (self.budget // 4) == 0:
            new_population_size = min(self.population_size + 5, 20 * self.dim)
            if new_population_size > self.population_size:
                additional_positions = np.random.rand(new_population_size - self.population_size, self.dim)
                self.positions = np.vstack((self.positions, additional_positions))
                self.velocities = np.vstack((self.velocities, np.zeros((new_population_size - self.population_size, self.dim))))
                self.pbest = np.vstack((self.pbest, additional_positions))
                self.pbest_scores = np.hstack((self.pbest_scores, np.full(new_population_size - self.population_size, float('inf'))))
                self.population_size = new_population_size

    def _adapt_beta(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.beta = min(1.0, self.beta + 0.05)

    def check_stagnation(self):
        if np.std(self.pbest_scores) < self.tol:
            self.positions += np.random.uniform(-0.5, 0.5, self.positions.shape)

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
                self.positions[i] = self.hybrid_local_search(self.positions[i], func)
                if self.evaluations >= self.budget:
                    break
            self._adapt_population()
            self._adapt_beta()
            self.check_stagnation()

        return self.gbest