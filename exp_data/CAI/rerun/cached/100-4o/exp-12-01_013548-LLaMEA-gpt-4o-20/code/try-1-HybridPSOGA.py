import numpy as np

class HybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.1
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, dim))
        self.best_positions = np.copy(self.positions)
        self.global_best_position = None
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]

    def _update_velocities_and_positions(self):
        for i in range(self.population_size):
            inertia = self.velocities[i]
            cognitive = np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i])
            social = np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            self.velocities[i] = inertia + cognitive + social
            self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

    def _crossover_and_mutate(self):
        for i in range(self.population_size):
            if np.random.rand() < self.crossover_rate:
                partner_index = np.random.randint(self.population_size)
                crossover_point = np.random.randint(1, self.dim)
                self.positions[i, :crossover_point] = self.best_positions[partner_index, :crossover_point]
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 1, self.dim)
                self.positions[i] += mutation_vector

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self._evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self._update_velocities_and_positions()
            self._crossover_and_mutate()
        
        return self.global_best_position