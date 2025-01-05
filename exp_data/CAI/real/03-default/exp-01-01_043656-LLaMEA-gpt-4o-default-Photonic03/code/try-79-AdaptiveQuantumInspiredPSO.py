import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(100, 10 * dim)
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7
        self.c1 = 2.05
        self.c2 = 2.05
        self.quantum_factor = 0.3
        self.epsilon = 1e-8  # small constant to avoid division by zero

    def quantum_jump(self, position, personal_best, global_best, convergence):
        delta = np.random.rand(self.dim)
        adaptive_qf = self.quantum_factor * (1 - convergence)
        new_position = position + adaptive_qf * (personal_best - position) * delta + adaptive_qf * (global_best - position) * (1 - delta)
        return new_position

    def adjust_population(self, eval_count):
        # Reduce population size as evaluations progress to focus on exploitation
        new_population_size = max(10, self.initial_population_size * (1 - eval_count / self.budget))
        self.population_size = int(new_population_size)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) * 0.1
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            self.adjust_population(eval_count)
            convergence = 1 - eval_count / self.budget
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best[i] - pop[i])
                                 + self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                trial = self.quantum_jump(pop[i], personal_best[i], global_best, convergence)
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