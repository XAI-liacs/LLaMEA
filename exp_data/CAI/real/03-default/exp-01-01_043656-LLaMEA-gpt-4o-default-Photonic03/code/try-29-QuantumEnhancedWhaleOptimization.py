import numpy as np

class QuantumEnhancedWhaleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.a_initial = 2.0
        self.a_final = 0.1
        self.b_initial = 1.5
        self.b_final = 0.5
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_superposition(self, position, best_position, eval_count):
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        delta = np.random.rand(self.dim)
        superposed_position = position + quantum_factor * (best_position - position) * delta
        return superposed_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            a = self.a_initial * (1 - eval_count / self.budget) + self.a_final * (eval_count / self.budget)
            b = self.b_initial * (1 - eval_count / self.budget) + self.b_final * (eval_count / self.budget)

            for i in range(self.population_size):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r

                if np.random.rand() < 0.5:
                    D = np.abs(C * global_best - pop[i])
                    pop[i] = global_best - A * D
                else:
                    l = np.random.uniform(-1, 1, size=self.dim)
                    D_prime = np.abs(global_best - pop[i])
                    pop[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best

                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                trial = self.quantum_superposition(pop[i], global_best, eval_count)
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