import numpy as np

class QuantumLevyAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.individuals = np.random.uniform(size=(self.population_size, dim))
        self.personal_best = self.individuals.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        adaptive_scale_factor = lambda evals: 0.6 + 0.2 * np.sin(2 * np.pi * evals / self.budget)
        adaptive_crossover_rate = lambda evals: 0.7 - 0.4 * (evals / self.budget)

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

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.individuals[indices[0]], self.individuals[indices[1]], self.individuals[indices[2]]

                F = adaptive_scale_factor(self.fitness_evaluations)
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = adaptive_crossover_rate(self.fitness_evaluations)
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, self.individuals[i])

                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                if trial_fitness < self.personal_best_fitness[i]:
                    self.individuals[i] = trial.copy()
                    self.personal_best[i] = trial.copy()
                    self.personal_best_fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial.copy()

            for i in range(self.population_size):
                levy_step = self.levy_flight(self.dim)
                if np.random.rand() < 0.3:
                    self.individuals[i] += levy_step * (self.individuals[i] - self.global_best)
                    self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

        return self.global_best