import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = np.array([0.5 * np.ones(self.dim), 0.5 * np.ones(self.dim)])
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7

    def qbit_to_solution(self):
        # Collapse q-bits to binary solution
        return np.random.rand(self.population_size, self.dim) < self.q_population[0]

    def evaluate_population(self, func, population):
        # Evaluate the objective function for the given population
        return np.array([func(indiv) for indiv in population])

    def update_q_population(self, best_indiv):
        # Update q-bit population according to the best solution found
        for i in range(self.dim):
            if best_indiv[i]:
                self.q_population[:, i] += self.mutation_rate
            else:
                self.q_population[:, i] -= self.mutation_rate
            self.q_population[:, i] = np.clip(self.q_population[:, i], 0, 1)

    def __call__(self, func):
        current_eval = 0
        best_solution = None
        best_score = float('inf')
        
        while current_eval < self.budget:
            binary_population = self.qbit_to_solution()
            population = binary_population * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
            scores = self.evaluate_population(func, population)
            current_eval += len(scores)
            
            # Find the best individual in the current population
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_solution = population[min_idx]
            
            # Update q-bit population towards the best individuals
            sorted_indices = np.argsort(scores)
            top_individuals = binary_population[sorted_indices[:self.population_size // 5]]
            for indiv in top_individuals:
                self.update_q_population(indiv)
                
            # Apply crossover and mutation to q_population
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                self.q_population[:, :crossover_point] = \
                    np.flip(self.q_population[:, :crossover_point], axis=1)

            mutation_mask = np.random.rand(*self.q_population.shape) < self.mutation_rate
            self.q_population[mutation_mask] = np.random.rand(np.sum(mutation_mask))
        
        return best_solution