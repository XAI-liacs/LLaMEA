import numpy as np
from scipy.optimize import minimize

class MultiPhaseEvolutionaryStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.bounds = None
        self.phase = 'exploration'

    def adaptive_initialization(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def periodicity_enforcement(self, solution):
        period_length = self.dim // 2
        for i in range(period_length):
            solution[i + period_length] = solution[i]
        return solution

    def transition_phase(self, generation):
        if generation < 0.7 * (self.budget // self.population_size):
            self.phase = 'exploration'
        else:
            self.phase = 'exploitation'

    def evolutionary_strategy(self, func):
        np.random.seed(42)
        population = self.adaptive_initialization(self.bounds.lb, self.bounds.ub)
        population_fitness = np.array([func(self.periodicity_enforcement(ind)) for ind in population])

        for generation in range(self.budget // self.population_size):
            self.transition_phase(generation)
            if self.phase == 'exploration':
                F = 0.8
                CR = 0.2
            else:
                F = 0.2
                CR = 0.8

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self.periodicity_enforcement(trial)

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]

        return best_individual

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.evolutionary_strategy(func)
        best_solution = self.local_optimization(func, best_global_solution)
        return best_solution