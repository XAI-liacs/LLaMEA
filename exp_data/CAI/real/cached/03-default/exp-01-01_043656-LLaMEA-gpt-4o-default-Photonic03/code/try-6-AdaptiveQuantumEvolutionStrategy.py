import numpy as np

class AdaptiveQuantumEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_rate = 0.1
        self.quantum_factor = 0.4
        self.sigma = 0.1

    def quantum_mutation(self, position, global_best):
        mutation = self.quantum_factor * (np.random.rand(self.dim) - 0.5) * 2
        new_position = position + mutation * (global_best - position)
        return new_position

    def adaptive_mutation(self, position, func_val, best_val):
        factor = np.exp(-(func_val - best_val) / (abs(best_val) + 1e-9))
        noise = self.sigma * factor * np.random.randn(self.dim)
        return position + noise

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        values = np.array([func(ind) for ind in pop])
        
        eval_count = self.population_size
        global_best_index = np.argmin(values)
        global_best = pop[global_best_index]
        global_best_value = values[global_best_index]
        
        while eval_count < self.budget:
            new_population = []
            new_values = []
            for i in range(self.population_size):
                candidate = self.quantum_mutation(pop[i], global_best)
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

                candidate_value = func(candidate)
                eval_count += 1

                if candidate_value < values[i]:
                    pop[i] = candidate
                    values[i] = candidate_value
                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                # Apply adaptive mutation
                candidate = self.adaptive_mutation(pop[i], values[i], global_best_value)
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

                candidate_value = func(candidate)
                eval_count += 1

                if candidate_value < values[i]:
                    pop[i] = candidate
                    values[i] = candidate_value
                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                new_population.append(pop[i])
                new_values.append(values[i])

                if eval_count >= self.budget:
                    break

            # Update population
            pop = np.array(new_population)
            values = np.array(new_values)
        
        return global_best