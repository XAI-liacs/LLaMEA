import numpy as np

class QuantumEnhancedDualSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1
        self.split_ratio = 0.5  # Ratio to split the dual swarms

    def quantum_update(self, position, personal_best, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = (position + personal_best) / 2 + quantum_factor * (global_best - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(pop)
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size
        split_index = int(self.population_size * self.split_ratio)
        swarm_a = np.arange(split_index)
        swarm_b = np.arange(split_index, self.population_size)

        while eval_count < self.budget:
            inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * \
                             (1 - eval_count / self.budget) + self.final_inertia_weight

            for swarm, global_best_current in zip([swarm_a, swarm_b], [global_best, global_best]):
                for i in swarm:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (inertia_weight * velocities[i]
                                     + self.c1 * r1 * (personal_best[i] - pop[i])
                                     + self.c2 * r2 * (global_best_current - pop[i]))
                    pop[i] += velocities[i]
                    pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                    trial = self.quantum_update(pop[i], personal_best[i], global_best_current, eval_count)
                    trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                    trial_value = func(trial)
                    eval_count += 1
                    if trial_value < personal_best_values[i]:
                        personal_best[i] = trial
                        personal_best_values[i] = trial_value
                        if trial_value < global_best_value:
                            global_best = trial
                            global_best_value = trial_value

                    if eval_count >= self.budget:
                        break

        return global_best