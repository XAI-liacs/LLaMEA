import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.mutation_factor = 0.8
        self.cross_over_rate = 0.9
        self.eval_count = 0

        # Initialize particles and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                fitness = func(self.positions[i])
                self.eval_count += 1
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i]
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i]

            # Update velocities and positions using PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Apply Differential Evolution strategy
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.cross_over_rate
                trial_vector = np.where(crossover, mutant_vector, self.positions[i])
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                if trial_fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = trial_fitness
                    self.pbest_positions[i] = trial_vector
                    if trial_fitness < self.gbest_score:
                        self.gbest_score = trial_fitness
                        self.gbest_position = trial_vector

        return self.gbest_position, self.gbest_score