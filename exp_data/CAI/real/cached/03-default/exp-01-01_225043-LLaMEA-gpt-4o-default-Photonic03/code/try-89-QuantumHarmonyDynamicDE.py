import numpy as np

class QuantumHarmonyDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.individuals = np.random.uniform(size=(self.population_size, dim))
        self.dynamic_memory = self.individuals.copy()
        self.global_best = None
        self.dynamic_memory_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def tune_parameters(self, evals):
        HMCR = 0.9 - 0.6 * (evals / self.budget)
        PAR = 0.1 + 0.4 * np.cos(np.pi * evals / self.budget)
        beta = 0.6 + 0.4 * np.sin(2 * np.pi * evals / self.budget)
        return HMCR, PAR, beta

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.individuals[i])
                self.fitness_evaluations += 1

                if fitness < self.dynamic_memory_fitness[i]:
                    self.dynamic_memory_fitness[i] = fitness
                    self.dynamic_memory[i] = self.individuals[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.individuals[i].copy()

            new_population = []
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                HMCR, PAR, beta = self.tune_parameters(self.fitness_evaluations)
                
                harmony_candidate = np.where(np.random.rand(self.dim) < HMCR, 
                                             self.dynamic_memory[np.random.choice(self.population_size)], 
                                             np.random.uniform(lower_bound, upper_bound, self.dim))
                
                if np.random.rand() < PAR:
                    harmony_candidate += beta * (np.random.rand(self.dim) - 0.5)

                harmony_candidate = np.clip(harmony_candidate, lower_bound, upper_bound)
                new_population.append(harmony_candidate)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.individuals[indices[0]], self.individuals[indices[1]], self.individuals[indices[2]]

                F = np.random.uniform(0.5, 0.8)
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.9 - 0.3 * (self.fitness_evaluations / self.budget)
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, new_population[i])

                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                if trial_fitness < self.dynamic_memory_fitness[i]:
                    self.individuals[i] = trial.copy()
                    self.dynamic_memory[i] = trial.copy()
                    self.dynamic_memory_fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial.copy()

        return self.global_best