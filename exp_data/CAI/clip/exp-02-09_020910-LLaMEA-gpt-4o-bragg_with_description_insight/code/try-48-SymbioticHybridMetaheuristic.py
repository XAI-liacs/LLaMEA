import numpy as np
from scipy.optimize import minimize

class SymbioticHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            new_population = np.empty_like(population)

            # Differential Evolution
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness > fitness[i]:
                    new_population[i] = trial
                else:
                    new_population[i] = population[i]

            # Symbiosis Phase: Mimicry-inspired Diversity Enhancement
            for j in range(self.population_size):
                if np.random.rand() < 0.5:
                    mimic_partner = new_population[np.random.randint(self.population_size)]
                    shift = np.random.randint(1, self.dim // 2)
                    new_population[j][:shift] = mimic_partner[-shift:]

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            self.evaluations += self.population_size

            # Selection
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[-self.population_size:]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            # Local refinement using BFGS
            for i in range(self.population_size):
                if np.random.rand() < 0.15 and self.evaluations < self.budget:
                    res = minimize(func, population[i], bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.success:
                        population[i] = res.x
                        fitness[i] = res.fun
                        self.evaluations += res.nfev

        best_index = np.argmax(fitness)
        return population[best_index]