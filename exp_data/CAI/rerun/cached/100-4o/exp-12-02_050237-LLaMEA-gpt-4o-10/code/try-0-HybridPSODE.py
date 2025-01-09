import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best = np.copy(self.population)
        self.global_best = self.population[np.random.randint(self.population_size)]
        self.fitness = np.full(self.population_size, np.inf)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.curr_evals = 0

    def evaluate(self, func):
        for i in range(self.population_size):
            if self.curr_evals >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.curr_evals += 1
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best[i] = self.population[i]
                self.personal_best_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.global_best_fitness:
                self.global_best = self.population[i]
                self.global_best_fitness = self.fitness[i]

    def update_velocities_and_positions(self):
        inertia_weight = 0.5
        cognitive_comp = 1.5
        social_comp = 1.5
        for i in range(self.population_size):
            if self.curr_evals >= self.budget:
                break
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = cognitive_comp * r1 * (self.personal_best[i] - self.population[i])
            social_velocity = social_comp * r2 * (self.global_best - self.population[i])
            self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

            # Differential Evolution Mutation
            if np.random.rand() < 0.5:
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[idxs]
                mutant_vector = x0 + 0.8 * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                if func(mutant_vector) < self.fitness[i]:
                    self.population[i] = mutant_vector
            else:
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.evaluate(func)
        while self.curr_evals < self.budget:
            self.update_velocities_and_positions()
            self.evaluate(func)
        return self.global_best, self.global_best_fitness