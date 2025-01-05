import numpy as np

class EnhancedAdaptiveMemoryPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.individuals = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best = self.individuals.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.cognitive_component = 1.5
        self.social_component = 2.5
        self.inertia_weight = 0.9
        self.memory_factor = 0.05

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v)**(1 / beta)
        return step

    def adaptive_mutation(self, individual, global_best, evals):
        mutation_rate = self.memory_factor + 0.45 * (1 - evals / self.budget)
        if np.random.rand() < mutation_rate:
            levy_step = self.levy_flight(self.dim)
            mutation = individual + levy_step * (global_best - individual)
            return np.clip(mutation, 0, 1)
        return individual

    def update_inertia_weight(self):
        self.inertia_weight = 0.9 - 0.5 * (self.fitness_evaluations / self.budget)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.individuals[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.individuals[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.individuals[i].copy()

            self.update_inertia_weight()

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best[i] - self.individuals[i])
                social_velocity = self.social_component * r2 * (self.global_best - self.individuals[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

                self.individuals[i] += self.velocities[i]
                self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

                self.individuals[i] = self.adaptive_mutation(self.individuals[i], self.global_best, self.fitness_evaluations)

        return self.global_best