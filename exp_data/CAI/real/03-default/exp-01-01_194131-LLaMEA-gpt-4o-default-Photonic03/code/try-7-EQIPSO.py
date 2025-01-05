import numpy as np

class EQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = self._chaotic_map_init(dim, self.population_size)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.phi = np.log(2)
        self.beta = 0.3
        self.evaluations = 0
        self.alpha = 0.5

    def _chaotic_map_init(self, dim, population_size):
        # Initialize positions using a logistic chaotic map
        x = np.random.rand(population_size, dim)
        r = 4.0  # Control parameter for logistic map
        for _ in range(100):  # Iterate to bring into chaotic regime
            x = r * x * (1 - x)
        return x

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (self.gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]
        
        # Quantum-inspired update with Levy flight
        if np.random.rand() < self.beta:
            new_pos += self.levy_flight()

        # Adaptive neighborhood exploration
        if np.random.rand() < self.alpha:
            neighborhood = np.random.normal(new_pos, 0.1, self.dim)
            new_pos = neighborhood

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

    def _adapt_parameters(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.beta = min(1.0, self.beta + 0.05)
            self.alpha = max(0.1, self.alpha - 0.05)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.positions
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
            self._adapt_population()
            self._adapt_parameters()

        return self.gbest