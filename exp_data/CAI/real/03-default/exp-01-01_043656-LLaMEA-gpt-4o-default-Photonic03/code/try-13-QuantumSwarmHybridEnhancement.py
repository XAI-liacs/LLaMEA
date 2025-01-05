import numpy as np

class QuantumSwarmHybridEnhancement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_tunneling_factor = 0.2
        self.diversity_injection_prob = 0.1

    def quantum_tunneling(self, position, global_best):
        return position + self.quantum_tunneling_factor * np.random.randn(self.dim) * (global_best - position)

    def diversity_injection(self, bounds):
        return np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

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
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best[i] - pop[i]) +
                                 self.social_coeff * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                if np.random.rand() < self.diversity_injection_prob:
                    pop[i] = self.diversity_injection(bounds)

                trial = self.quantum_tunneling(pop[i], global_best)
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