import numpy as np
from scipy.optimize import minimize

class AdaptiveDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop_size = max(5, self.budget // 20)
        F = 0.8
        CR = 0.9
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_score = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation and crossover
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation of trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution if improved
                    if trial_fitness < best_score:
                        best_solution = trial
                        best_score = trial_fitness

                        # Local optimization using L-BFGS-B
                        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': self.budget - evaluations})
                        evaluations += result.nfev
                        if result.fun < best_score:
                            best_score = result.fun
                            best_solution = result.x

            # Dynamic adjustment based on fitness landscape
            F = 0.5 + 0.5 * (1 - best_score / max(fitness))  # Modified line
            CR = 0.7 + 0.3 * (best_score / max(fitness))  # Modified line

            # Dynamically adjust population size based on evaluations
            pop_size = max(5, (self.budget - evaluations) // 20)

        return best_solution