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
        convergence_speed = np.mean(np.abs(fitness - np.min(fitness)))
        F = 0.5 + 0.5 * np.exp(-convergence_speed)
        diversity = np.std(pop, axis=0).mean()
        CR = 0.9 * np.exp(-diversity)
        weight_factor = 0.5 + 0.5 * np.exp(-convergence_speed) * np.random.rand()  # Dynamic weight adjustment with randomness
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + weight_factor * (b - c), lb, ub)
            trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
            trial = self.enforce_symmetry(trial, lb, ub)
            trial_fitness = self.func(trial)
            if trial_fitness < fitness[i]:
                pop[i], fitness[i] = trial, trial_fitness

    def enforce_symmetry(self, solution, lb, ub):
        midpoint = len(solution) // 2
        for i in range(midpoint):
            avg = (solution[i] + solution[-(i + 1)]) / 2
            solution[i] = solution[-(i + 1)] = np.clip(avg, lb[i], ub[i])
        return solution

    def local_optimization(self, best_solution, lb, ub):
        res = minimize(self.func, best_solution, bounds=list(zip(lb, ub)),
                       method='L-BFGS-B', options={'maxiter': self.budget // 10})
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