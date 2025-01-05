import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor_initial = 0.8
        self.mutation_factor_final = 0.5
        self.crossover_rate = 0.9
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_perturbation(self, ind, best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_ind = ind + quantum_factor * (best - ind) * delta
        return new_ind

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        pop_values = np.array([func(ind) for ind in pop])
        best = pop[np.argmin(pop_values)]
        best_value = pop_values.min()
        
        eval_count = self.population_size

        while eval_count < self.budget:
            mutation_factor = self.mutation_factor_initial * (1 - eval_count / self.budget) + self.mutation_factor_final * (eval_count / self.budget)

            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = pop[indices]

                mutant = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.array([mutant[j] if np.random.rand() < self.crossover_rate else pop[i][j] for j in range(self.dim)])
                
                trial_quantum = self.quantum_perturbation(trial, best, eval_count)
                trial_quantum = np.clip(trial_quantum, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(trial_quantum)
                eval_count += 1

                if trial_value < pop_values[i]:
                    pop[i] = trial_quantum
                    pop_values[i] = trial_value
                    if trial_value < best_value:
                        best = trial_quantum
                        best_value = trial_value

                if eval_count >= self.budget:
                    break

        return best