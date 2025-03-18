import numpy as np
from scipy.optimize import minimize

class DynamicScalingSymmetryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.fitness = np.inf * np.ones(self.population_size)
        self.bounds = None

    def initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        return pop

    def differential_evolution_step(self, pop, fitness, lb, ub):
        F = 0.6  # Adjusted scaling factor
        Cr = 0.9 - 0.5 * np.std(fitness) / np.mean(fitness)  # Refine crossover rate adaptation
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)
            trial = np.where(np.random.rand(self.dim) < Cr, mutant, pop[i])
            trial = self.enforce_symmetry(trial, lb, ub)
            trial = self.enforce_periodicity(trial, lb, ub)
            trial_fitness = self.func(trial)
            if trial_fitness < fitness[i]:
                pop[i], fitness[i] = trial, trial_fitness

    def enforce_symmetry(self, solution, lb, ub):
        midpoint = len(solution) // 2
        for i in range(midpoint):
            avg = (solution[i] + solution[-(i + 1)]) / 2
            solution[i] = solution[-(i + 1)] = np.clip(avg, lb[i], ub[i])
        return solution

    def enforce_periodicity(self, solution, lb, ub):
        diversity_factor = np.std(solution) / np.mean(solution)
        if diversity_factor < 0.15:  # Adjusted threshold for periodicity enforcement
            solution = np.clip(np.mean(solution) * np.ones(self.dim), lb, ub)
        else:
            phase_shift = np.mean(solution[:self.dim // 2]) - np.mean(solution[self.dim // 2:])
            solution[:self.dim // 2] -= phase_shift / 2
            solution[self.dim // 2:] += phase_shift / 2
        return solution

    def local_optimization(self, best_solution, lb, ub):
        res = minimize(self.func, best_solution, bounds=list(zip(lb, ub)),
                       method='L-BFGS-B', options={'maxiter': min(self.budget // 6, 150)})  # Adjusted maxiter for local search
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
                evaluations += min(self.budget // 8, 100)  # Adjusted evaluations increment

        return pop[np.argmin(self.fitness)]