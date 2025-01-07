import numpy as np

class BioInspiredEnvironmentalAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_step_size = 0.05
        self.final_step_size = 0.01
        self.local_weight_initial = 0.7
        self.local_weight_final = 0.3
        self.global_weight_initial = 0.3
        self.global_weight_final = 0.7

    def environment_update(self, position, local_best, global_best, eval_count):
        alpha = eval_count / self.budget
        step_size = (self.initial_step_size * (1 - alpha) + self.final_step_size * alpha)
        local_weight = (self.local_weight_initial * (1 - alpha) + self.local_weight_final * alpha)
        global_weight = (self.global_weight_initial * alpha + self.global_weight_final * (1 - alpha))
        
        disturbance = np.random.normal(0, step_size, self.dim)
        new_position = (1 - local_weight - global_weight) * position + \
                       local_weight * local_best + \
                       global_weight * global_best + disturbance
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        local_best = pop.copy()
        local_best_values = np.array([func(ind) for ind in pop])
        global_best = local_best[np.argmin(local_best_values)]
        global_best_value = local_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                candidate = self.environment_update(pop[i], local_best[i], global_best, eval_count)
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                
                candidate_value = func(candidate)
                eval_count += 1
                if candidate_value < local_best_values[i]:
                    local_best[i] = candidate
                    local_best_values[i] = candidate_value
                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                if eval_count >= self.budget:
                    break

        return global_best