import numpy as np

class BioInspiredDispersiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.dispersal_factor_initial = 0.5
        self.dispersal_factor_final = 0.1

    def dispersive_update(self, position, global_best, eval_count):
        dispersal_factor = self.dispersal_factor_initial * (1 - (eval_count / self.budget)) + self.dispersal_factor_final * (eval_count / self.budget)
        random_direction = np.random.uniform(-1, 1, self.dim)
        dispersive_position = position + dispersal_factor * random_direction
        return dispersive_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(pop)
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * \
                             (1 - eval_count / self.budget) + self.final_inertia_weight

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best[i] - pop[i])
                                 + self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                dispersive_trial = self.dispersive_update(pop[i], global_best, eval_count)
                dispersive_trial = np.clip(dispersive_trial, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(dispersive_trial)
                eval_count += 1
                if trial_value < personal_best_values[i]:
                    personal_best[i] = dispersive_trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = dispersive_trial
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return global_best