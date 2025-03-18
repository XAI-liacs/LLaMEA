import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, int(0.02 * budget))  
        self.mutation_rate = 0.1
        self.q = np.full((self.population_size, dim), 0.5)  
        self.elite_fraction = 0.1

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.elites = []

    def measure(self):
        return np.where(np.random.rand(*self.q.shape) < self.q, 1, 0)

    def update_quantum_probabilities(self, best_solution):
        learning_rate = 0.05
        adaptive_factor = np.random.uniform(0.9, 1.1)  # changed line to introduce adaptive adjustment
        for i in range(self.population_size):
            for j in range(self.dim):
                if self.population[i, j] == best_solution[j]:
                    self.q[i, j] = min(0.9, self.q[i, j] + learning_rate * adaptive_factor)
                else:
                    self.q[i, j] = max(0.1, self.q[i, j] - learning_rate * adaptive_factor)

    def evolutionary_process(self, lb, ub):
        offspring = np.clip(np.random.normal(loc=self.q, scale=0.1), 0, 1)
        real_values = lb + (ub - lb) * (offspring / (self.q.shape[1] - 1))
        fitness_variance = np.var(self.fitness)
        
        best_fitness = np.min(self.fitness)
        dynamic_mutation_rate = self.mutation_rate + 0.1 * fitness_variance * (1 - best_fitness)

        mutation_mask = np.random.rand(*real_values.shape) < dynamic_mutation_rate
        real_values[mutation_mask] += np.random.normal(0, 0.1, real_values[mutation_mask].shape)
        np.clip(real_values, lb, ub, out=real_values)  

        return real_values

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.evolutionary_process(lb, ub)
            new_fitness = np.apply_along_axis(func, 1, new_population)
            eval_count += len(new_fitness)

            combined_population = np.vstack((self.population, new_population))
            combined_fitness = np.hstack((self.fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            
            self.population = combined_population[best_indices]
            self.fitness = combined_fitness[best_indices]
            
            best_idx = np.argmin(self.fitness)
            best_solution = self.population[best_idx]
            self.update_quantum_probabilities(best_solution)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]