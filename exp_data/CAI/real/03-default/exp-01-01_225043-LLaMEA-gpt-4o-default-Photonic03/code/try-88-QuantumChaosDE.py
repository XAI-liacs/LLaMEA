import numpy as np

class QuantumChaosDE:
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

    def chaos_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        adaptive_scale_factor = lambda evals: 0.5 + 0.3 * np.sin(2 * np.pi * evals / self.budget)
        adaptive_crossover_rate = lambda evals: 0.8 - 0.5 * (evals / self.budget)

        chaos = np.random.uniform(size=self.population_size)

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

                indices = np.random.choice(self.population_size, 5, replace=False)
                tournament = indices[np.argsort([self.personal_best_fitness[j] for j in indices])[:3]]
                a, b, c = self.individuals[tournament[0]], self.individuals[tournament[1]], self.individuals[tournament[2]]

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
                chaos[i] = self.chaos_map(chaos[i])
                levy_step = chaos[i] * (self.individuals[i] - self.global_best)
                if np.random.rand() < 0.3:
                    self.individuals[i] += levy_step
                    self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

        return self.global_best