import numpy as np
from scipy.optimize import minimize

class CollaborativeCoevolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpop_size = 10
        self.num_subpops = 2
        self.bounds = None

    def initialize_subpopulations(self, lb, ub):
        subpopulations = [np.random.uniform(lb, ub, (self.subpop_size, self.dim // self.num_subpops)) for _ in range(self.num_subpops)]
        return subpopulations

    def periodicity_induced_mutation(self, subpop, dim):
        period_length = dim // 2
        for ind in subpop:
            for i in range(period_length):
                ind[i + period_length] = np.mean(ind[:period_length])
        return subpop

    def crossover(self, parent1, parent2, CR):
        dim = len(parent1)
        cross_points = np.random.rand(dim) < CR
        child = np.where(cross_points, parent1, parent2)
        return child

    def collaborative_evolution(self, func):
        np.random.seed(42)
        subpopulations = self.initialize_subpopulations(self.bounds.lb, self.bounds.ub)
        subpop_fitness = [np.array([func(np.concatenate(subpopulations)) for _ in subpop]) for subpop in subpopulations]

        for generation in range(self.budget // (self.subpop_size * self.num_subpops)):
            F = 0.8 + 0.2 * np.sin(2 * np.pi * generation / self.budget)
            CR = 0.8 + 0.2 * np.cos(2 * np.pi * generation / self.budget)
            for sp_idx, subpop in enumerate(subpopulations):
                for i in range(self.subpop_size):
                    indices = [idx for idx in range(self.subpop_size) if idx != i]
                    a, b, c = subpop[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)
                    trial = self.crossover(mutant, subpop[i], CR)
                    trial = self.periodicity_induced_mutation(trial, self.dim // self.num_subpops)

                    trial_fitness = func(np.concatenate(subpopulations))
                    if trial_fitness < subpop_fitness[sp_idx][i]:
                        subpop[i] = trial
                        subpop_fitness[sp_idx][i] = trial_fitness

        best_combined_solution = np.concatenate([subpop[np.argmin(fitness)] for subpop, fitness in zip(subpopulations, subpop_fitness)])
        return best_combined_solution

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_collaborative_solution = self.collaborative_evolution(func)
        best_solution = self.local_optimization(func, best_collaborative_solution)
        return best_solution