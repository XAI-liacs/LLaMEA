import numpy as np

class AdaptiveDifferentialRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        evaluations = 1
        no_improvement_counter = 0
        population = [best_solution]

        while evaluations < self.budget:
            diversity = np.std(population, axis=0).mean() / (ub - lb).mean()  # New line
            exploration_factor = np.random.uniform(0.05, 0.2) * (1 - evaluations / self.budget) ** 2
            inertia_weight = 0.9 * np.exp(-evaluations / (0.1 * self.budget)) * (1 + diversity)  # Modified line
            if np.random.rand() < 0.15:
                trial_solution = np.random.uniform(lb, ub, self.dim)
            else:
                trial_solution = best_solution + inertia_weight * np.random.uniform(-1, 1, self.dim) * (ub - lb) * exploration_factor
            trial_solution = np.clip(trial_solution, lb, ub)
            trial_value = func(trial_solution)
            evaluations += 1

            if trial_value < best_value:
                best_solution = trial_solution
                best_value = trial_value
                no_improvement_counter = 0
                population.append(trial_solution)  # New line
            else:
                no_improvement_counter += 1
                if no_improvement_counter > self.budget * 0.05:
                    trial_solution = np.random.uniform(lb, ub, self.dim)
                    if no_improvement_counter > self.budget * 0.1:  # New line
                        best_solution = np.random.uniform(lb, ub, self.dim)  # New line
                        best_value = func(best_solution)  # New line
                        population = [best_solution]  # New line

        return best_solution