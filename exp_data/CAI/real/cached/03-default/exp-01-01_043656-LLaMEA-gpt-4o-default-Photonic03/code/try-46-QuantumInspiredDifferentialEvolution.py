import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor_initial = 0.8
        self.mutation_factor_final = 0.3
        self.crossover_rate = 0.9

    def quantum_superposition(self, parent1, parent2, global_best, eval_count):
        lambda_factor = (eval_count / self.budget)
        mutation_factor = self.mutation_factor_initial * (1 - lambda_factor) + self.mutation_factor_final * lambda_factor
        weight = np.random.uniform(0, 1, self.dim)
        trial_vector = weight * parent1 + (1 - weight) * parent2 + mutation_factor * (global_best - parent1)
        return trial_vector

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness_values = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness_values)]
        global_best_value = fitness_values.min()
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                r1, r2 = indices[0], indices[1]
                
                trial_vector = self.quantum_superposition(pop[r1], pop[r2], global_best, eval_count)
                trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(trial_vector)
                eval_count += 1
                
                if trial_value < fitness_values[i]:
                    pop[i] = trial_vector
                    fitness_values[i] = trial_value
                    
                    if trial_value < global_best_value:
                        global_best = trial_vector
                        global_best_value = trial_value
                
                if eval_count >= self.budget:
                    break
        
        return global_best