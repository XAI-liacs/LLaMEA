import numpy as np

class AdaptiveQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.crossover_rate_initial = 0.9
        self.crossover_rate_final = 0.4
        self.quantum_mutation_rate = 0.1

    def quantum_mutation(self, individual, global_best, eval_count):
        lambda_factor = (eval_count / self.budget)  # Adaptive factor
        mutation_vector = np.random.rand(self.dim)
        new_individual = individual + self.quantum_mutation_rate * lambda_factor * (global_best - individual) * mutation_vector
        return new_individual

    def crossover(self, parent1, parent2, eval_count):
        lambda_factor = (eval_count / self.budget)
        crossover_rate = self.crossover_rate_initial * (1 - lambda_factor) + self.crossover_rate_final * lambda_factor
        mask = np.random.rand(self.dim) < crossover_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness_values = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness_values)]
        global_best_value = fitness_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parents = np.random.choice(self.population_size, 2, replace=False, p=None)
                parent1, parent2 = pop[parents[0]], pop[parents[1]]
                
                offspring1 = self.crossover(parent1, parent2, eval_count)
                offspring2 = self.crossover(parent2, parent1, eval_count)
                
                offspring1 = self.quantum_mutation(offspring1, global_best, eval_count)
                offspring2 = self.quantum_mutation(offspring2, global_best, eval_count)
                
                offspring1 = np.clip(offspring1, bounds[:, 0], bounds[:, 1])
                offspring2 = np.clip(offspring2, bounds[:, 0], bounds[:, 1])
                
                new_population.extend([offspring1, offspring2])

            new_population = np.array(new_population)
            new_fitness_values = np.array([func(ind) for ind in new_population])
            eval_count += self.population_size
            
            # Combine and select the best individuals
            combined_pop = np.vstack((pop, new_population))
            combined_fitness = np.concatenate((fitness_values, new_fitness_values))
            
            best_indices = np.argpartition(combined_fitness, self.population_size)[:self.population_size]
            pop = combined_pop[best_indices]
            fitness_values = combined_fitness[best_indices]
            
            current_best_value = fitness_values.min()
            if current_best_value < global_best_value:
                global_best_value = current_best_value
                global_best = pop[np.argmin(fitness_values)]

        return global_best