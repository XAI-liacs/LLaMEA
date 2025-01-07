import numpy as np

class QuantumInspiredGravitationalWaveOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_gravitation = 0.9
        self.final_gravitation = 0.4
        self.quantum_tunneling_prob = 0.2

    def quantum_tunnel(self, position, global_best, eval_count):
        lambda_factor = (eval_count / self.budget)
        scale = (1 - lambda_factor) * self.initial_gravitation + lambda_factor * self.final_gravitation
        if np.random.rand() < self.quantum_tunneling_prob:
            perturbation = np.random.randn(self.dim) * scale
            return position + perturbation
        return position

    def gravitational_wave_propagate(self, pop, global_best, gravitation):
        for i in range(self.population_size):
            r = np.random.rand(self.dim)
            pop[i] += gravitation * r * (global_best - pop[i])
            pop[i] = np.clip(pop[i], self.bounds[:, 0], self.bounds[:, 1])

    def __call__(self, func):
        self.bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            gravitation = (self.initial_gravitation - self.final_gravitation) * \
                          (1 - eval_count / self.budget) + self.final_gravitation

            self.gravitational_wave_propagate(pop, global_best, gravitation)

            for i in range(self.population_size):
                trial = self.quantum_tunnel(pop[i], global_best, eval_count)
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
                
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