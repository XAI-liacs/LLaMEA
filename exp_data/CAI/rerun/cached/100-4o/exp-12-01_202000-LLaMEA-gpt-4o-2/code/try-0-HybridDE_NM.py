import numpy as np
from scipy.optimize import minimize

class HybridDE_NM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim)
        )
        self.num_evaluations = 0

    def differential_evolution(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.num_evaluations += self.population_size

        for i in range(self.population_size):
            if self.num_evaluations >= self.budget:
                break
            candidates = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
            a, b, c = self.population[candidates]
            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

            crossover = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True

            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.num_evaluations += 1

            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                self.population[i] = trial
        return fitness

    def nelder_mead(self, func, best_individual):
        if self.num_evaluations >= self.budget:
            return best_individual
        res = minimize(
            func, best_individual, method='Nelder-Mead', 
            bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
            options={'maxfev': self.budget - self.num_evaluations, 'disp': False}
        )
        self.num_evaluations += res.nfev
        return res.x

    def __call__(self, func):
        while self.num_evaluations < self.budget:
            fitness = self.differential_evolution(func)
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            best_individual = self.nelder_mead(func, best_individual)
            self.population[best_idx] = best_individual
        return self.population[np.argmin([func(ind) for ind in self.population])]