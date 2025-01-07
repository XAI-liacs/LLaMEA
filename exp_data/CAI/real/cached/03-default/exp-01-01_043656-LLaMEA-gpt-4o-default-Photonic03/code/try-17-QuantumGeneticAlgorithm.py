import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_rate_initial = 0.1
        self.mutation_rate_final = 0.001
        self.quantum_factor_initial = 0.05
        self.quantum_factor_final = 0.001

    def quantum_mutation(self, individual, eval_count):
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        mutation_vector = np.random.normal(0, quantum_factor, self.dim)
        return individual + mutation_vector

    def adaptive_mutation_rate(self, eval_count):
        return self.mutation_rate_initial * (1 - eval_count / self.budget) + self.mutation_rate_final * (eval_count / self.budget)

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = fitness.min()

        while eval_count < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parents_indices = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
                child1, child2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                
                mutation_rate = self.adaptive_mutation_rate(eval_count)
                if np.random.rand() < mutation_rate:
                    child1 = self.quantum_mutation(child1, eval_count)
                if np.random.rand() < mutation_rate:
                    child2 = self.quantum_mutation(child2, eval_count)

                new_population.extend([child1, child2])
            
            new_population = np.array(new_population)
            new_population = np.clip(new_population, bounds[:, 0], bounds[:, 1])

            new_fitness = np.array([func(ind) for ind in new_population])
            eval_count += self.population_size

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))

            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            best_candidate_index = np.argmin(fitness)
            if fitness[best_candidate_index] < best_fitness:
                best_solution = population[best_candidate_index]
                best_fitness = fitness[best_candidate_index]

        return best_solution