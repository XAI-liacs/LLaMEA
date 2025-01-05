import numpy as np

class QuantumEnhancedAdaptivePSO_MSC:
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
        self.num_swarms = 3  # Increase collaboration by using multiple swarms

    def quantum_update(self, position, personal_best, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)  # Adaptive quantum factor
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = (position + personal_best) / 2 + quantum_factor * (global_best - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        # Initialize multiple swarms
        swarms = [
            np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            for _ in range(self.num_swarms)
        ]
        velocities = [np.zeros_like(swarm) for swarm in swarms]
        personal_bests = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_bests = [personal_bests[i][np.argmin(personal_best_values[i])] for i in range(self.num_swarms)]
        global_best_values = [values.min() for values in personal_best_values]

        eval_count = self.population_size * self.num_swarms

        while eval_count < self.budget:
            for swarm_idx in range(self.num_swarms):
                inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * \
                                 (1 - eval_count / self.budget) + self.final_inertia_weight
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[swarm_idx][i] = (inertia_weight * velocities[swarm_idx][i]
                                                + self.c1 * r1 * (personal_bests[swarm_idx][i] - swarms[swarm_idx][i])
                                                + self.c2 * r2 * (global_bests[swarm_idx] - swarms[swarm_idx][i]))
                    swarms[swarm_idx][i] += velocities[swarm_idx][i]
                    swarms[swarm_idx][i] = np.clip(swarms[swarm_idx][i], bounds[:, 0], bounds[:, 1])

                    trial = self.quantum_update(swarms[swarm_idx][i], personal_bests[swarm_idx][i], global_bests[swarm_idx], eval_count)
                    trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                    trial_value = func(trial)
                    eval_count += 1
                    if trial_value < personal_best_values[swarm_idx][i]:
                        personal_bests[swarm_idx][i] = trial
                        personal_best_values[swarm_idx][i] = trial_value
                        if trial_value < global_best_values[swarm_idx]:
                            global_bests[swarm_idx] = trial
                            global_best_values[swarm_idx] = trial_value

                    if eval_count >= self.budget:
                        break

            # Collaborate by sharing best solutions between swarms
            overall_global_best_value = min(global_best_values)
            for swarm_idx in range(self.num_swarms):
                if global_best_values[swarm_idx] > overall_global_best_value:
                    global_bests[swarm_idx] = global_bests[np.argmin(global_best_values)]
                    global_best_values[swarm_idx] = overall_global_best_value

        best_swarm_idx = np.argmin(global_best_values)
        return global_bests[best_swarm_idx]