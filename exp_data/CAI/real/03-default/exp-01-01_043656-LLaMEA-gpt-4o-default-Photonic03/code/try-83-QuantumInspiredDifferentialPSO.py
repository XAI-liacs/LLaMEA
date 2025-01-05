import numpy as np

class QuantumInspiredDifferentialPSO:
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
        self.differential_weight = 0.8
        self.crossover_prob = 0.9

    def quantum_update(self, position, personal_best, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = (position + personal_best) / 2 + quantum_factor * (global_best - position) * delta
        return new_position

    def differential_mutation(self, pop, idx, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.differential_weight * (b - c), bounds[:, 0], bounds[:, 1])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

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

                # Perform differential mutation and crossover
                mutant = self.differential_mutation(pop, i, bounds)
                offspring = self.crossover(pop[i], mutant)
                offspring = np.clip(offspring, bounds[:, 0], bounds[:, 1])

                trial_value = func(offspring)
                eval_count += 1
                if trial_value < personal_best_values[i]:
                    personal_best[i] = offspring
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = offspring
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return global_best