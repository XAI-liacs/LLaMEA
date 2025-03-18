import numpy as np
from scipy.optimize import minimize

class DynamicAdaptiveQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.bounds = None

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adaptive_mutation(self, population, diversity):
        F_base = 0.5
        return F_base * (1 + (1 - diversity) / 2)

    def quantum_superposition_crossover(self, a, b, c):
        alpha = np.random.rand(self.dim)
        beta = np.sqrt(1 - alpha**2)
        return alpha * a + beta * b + (1 - alpha - beta) * c

    def differential_evolution(self, func):
        np.random.seed(42)
        population = self.initialize_population(self.bounds.lb, self.bounds.ub)
        population_fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget // self.population_size):
            diversity = np.std(population, axis=0).mean()
            F = self.adaptive_mutation(population, diversity)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(self.quantum_superposition_crossover(a, b, c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < 0.9
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]

        return best_individual

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='SLSQP', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.differential_evolution(func)
        best_solution = self.local_optimization(func, best_global_solution)
        return best_solution