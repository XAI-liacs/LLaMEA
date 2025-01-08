import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = np.array([0.5 * np.ones(self.dim), 0.5 * np.ones(self.dim)])
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.min_population_size = 10  # New line to define minimum population size

    def qbit_to_solution(self):
        return np.random.rand(self.population_size, self.dim) < self.q_population[0]

    def evaluate_population(self, func, population):
        return np.array([func(indiv) for indiv in population])

    def update_q_population(self, best_indiv):
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
            
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_solution = population[min_idx]
            
            sorted_indices = np.argsort(scores)
            top_individuals = binary_population[sorted_indices[:self.population_size // 5]]
            elite_individual = binary_population[min_idx]
            self.update_q_population(elite_individual)
            
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                self.q_population[:, :crossover_point] = \
                    np.flip(self.q_population[:, :crossover_point], axis=1)

            self.q_population += self.mutation_rate * np.random.uniform(-1, 1, self.q_population.shape)
            
            # Introduce adaptive population resizing
            self.population_size = max(self.min_population_size, int(self.population_size * 0.9)) # Adaptive resizing

        return best_solution