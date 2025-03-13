import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, iters, pop_size=50, F=0.8, CR=0.9):
        pop_size = max(4, pop_size // 2)
        population = np.random.rand(pop_size, self.dim)
        population *= (bounds.ub - bounds.lb)
        population += bounds.lb
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        for generation in range(iters):
            if self.evaluations >= self.budget:
                break
            F = 0.5 + (0.5 * np.random.rand())  # Changed for improved adaptive F control
            CR = 0.8 + (0.1 * np.random.rand())  # Changed for improved adaptive CR control
            best_idx = np.argmin(fitness)  # Elitism: Identify the best individual
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
            if pop_size < 100:  # Dynamically increase population size
                pop_size = min(pop_size + 1, 100)
                new_individual = np.random.rand(1, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
                population = np.vstack([population, new_individual])
                fitness = np.append(fitness, func(new_individual))

            # New line: Perform local search on the best individual every few generations
            if generation % 5 == 0:
                best_individual = population[best_idx]
                refined_solution, refined_cost = self.local_refinement(func, bounds, best_individual)
                if refined_cost < fitness[best_idx]:
                    population[best_idx] = refined_solution
                    fitness[best_idx] = refined_cost

        return population[best_idx], np.min(fitness)

    def local_refinement(self, func, bounds, x0):
        result = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        iterations = self.budget // 10
        global_solution, global_cost = self.differential_evolution(func, bounds, iters=iterations)
        local_solution, local_cost = self.local_refinement(func, bounds, global_solution)
        
        if local_cost < global_cost:
            return local_solution
        else:
            return global_solution