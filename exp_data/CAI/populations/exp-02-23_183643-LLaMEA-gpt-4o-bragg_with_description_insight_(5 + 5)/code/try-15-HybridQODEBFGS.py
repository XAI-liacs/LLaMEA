import numpy as np
from scipy.optimize import minimize

class HybridQODEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.fitness = np.inf * np.ones(self.population_size)
        self.bounds = None

    def quasi_opposition(self, x, lb, ub):
        return lb + ub - x

    def initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = self.quasi_opposition(pop, lb, ub)
        return np.concatenate((pop, opp_pop), axis=0)

    def differential_evolution_step(self, pop, fitness, lb, ub):
        diversity = np.std(pop, axis=0).mean()
        F = 0.5 + 0.3 * (diversity / (ub - lb).mean())  # Adaptive mutation factor
        for i in range(self.population_size):
            indices = [idx for idx in range(2 * self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)
            trial = np.where(np.random.rand(self.dim) < 0.9, mutant, pop[i])
            trial_fitness = self.func(trial)
            if trial_fitness < fitness[i]:
                pop[i], fitness[i] = trial, trial_fitness

    def local_optimization(self, best_solution, lb, ub):
        res = minimize(self.func, best_solution, bounds=list(zip(lb, ub)), 
                       method='L-BFGS-B', options={'maxiter': self.budget//10})
        return res.x, res.fun

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds
        lb, ub = self.bounds.lb, self.bounds.ub
        
        pop = self.initialize_population(lb, ub)
        self.fitness = np.array([self.func(ind) for ind in pop])
        evaluations = len(pop)

        while evaluations < self.budget:
            self.differential_evolution_step(pop, self.fitness, lb, ub)
            evaluations += self.population_size

            if evaluations < self.budget:
                best_index = np.argmin(self.fitness)
                best_solution, best_fitness = self.local_optimization(pop[best_index], lb, ub)
                if best_fitness < self.fitness[best_index]:
                    pop[best_index], self.fitness[best_index] = best_solution, best_fitness
                evaluations += self.budget // 10

        return pop[np.argmin(self.fitness)]