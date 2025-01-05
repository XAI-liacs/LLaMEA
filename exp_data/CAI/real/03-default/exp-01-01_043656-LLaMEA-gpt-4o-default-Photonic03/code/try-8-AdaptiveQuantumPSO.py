import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.inertia_weight = 0.9   # Start with a high inertia weight
        self.inertia_min = 0.4      # Lower bound for inertia
        self.inertia_decay = 0.99   # Decay factor for inertia
        self.c1 = 1.5               # Cognitive component
        self.c2 = 1.5               # Social component
        self.c1_decay = 0.995       # Decay factor for cognitive component
        self.c2_growth = 1.005      # Growth factor for social component
        self.quantum_factor = 0.3

    def quantum_update(self, position, personal_best, global_best):
        delta = np.random.rand(self.dim)
        new_position = (position + personal_best) / 2 + self.quantum_factor * (global_best - position) * delta
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

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Adaptive inertia weight and learning factors
                self.inertia_weight = max(self.inertia_min, self.inertia_weight * self.inertia_decay)
                self.c1 *= self.c1_decay
                self.c2 *= self.c2_growth
                
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best[i] - pop[i])
                                 + self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                trial = self.quantum_update(pop[i], personal_best[i], global_best)
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