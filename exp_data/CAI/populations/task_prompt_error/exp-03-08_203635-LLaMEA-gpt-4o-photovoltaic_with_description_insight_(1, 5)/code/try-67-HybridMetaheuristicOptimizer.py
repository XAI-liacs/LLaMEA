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

        velocity = np.random.rand(pop_size, self.dim) * 0.1  # Initialize velocity for PSO
        best_personal = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        for generation in range(iters):
            if self.evaluations >= self.budget:
                break
            F = 0.5 + (0.4 * np.abs(np.sin(generation + np.pi/4)))
            CR = 0.8 + (0.15 * np.abs(np.cos(generation + np.pi/3)))
            best_idx = np.argmin(fitness)
            valid_pop_size = len(population)
            global_best = population[best_idx]

            for i in range(valid_pop_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = 0.5 * velocity[i] + r1 * (best_personal[i] - population[i]) + r2 * (global_best - population[i])  # Update velocity
                trial = population[i] + velocity[i]  # Move particles
                trial = np.clip(trial, bounds.lb, bounds.ub)

                mutant = np.clip(population[i] + F * (trial - global_best), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < personal_best_fitness[i]:  # Update personal best
                        best_personal[i] = trial
                        personal_best_fitness[i] = trial_fitness

            if generation % 3 == 0:  # Adjust local refinement frequency
                best_individual = population[best_idx]
                for _ in range(2):
                    refined_solution, refined_cost = self.local_refinement(func, bounds, best_individual)
                    if refined_cost < fitness[best_idx]:
                        population[best_idx] = refined_solution
                        fitness[best_idx] = refined_cost
            pop_size = min(pop_size + 2, 100)
            new_individual = np.random.rand(1, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
            new_fitness = np.mean([func(new_individual[0]) for _ in range(3)]) 
            fitness = np.append(fitness, new_fitness)
            population = np.vstack([population, new_individual])

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