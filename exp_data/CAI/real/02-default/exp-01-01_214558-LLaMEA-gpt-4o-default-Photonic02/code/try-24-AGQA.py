import numpy as np

class AGQA:
    def __init__(self, budget, dim, population_size=20, crossover_prob=0.8, mutation_prob=0.1, quantum_prob=0.2, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.quantum_prob = quantum_prob
        self.diversity_threshold = diversity_threshold
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_global_position = None
        best_global_value = float('inf')

        while self.evaluations < self.budget:
            diversity = self.calculate_diversity(population)
            dynamic_quantum_prob = self.quantum_prob * (1 + (self.diversity_threshold - diversity))
            selected_population = self.selection(population, func)

            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size and np.random.rand() < self.crossover_prob:
                    offspring1, offspring2 = self.crossover(selected_population[i], selected_population[i+1])
                    selected_population[i], selected_population[i+1] = offspring1, offspring2

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_prob:
                    selected_population[i] = self.mutation(selected_population[i], lb, ub)

                if np.random.rand() < dynamic_quantum_prob:
                    selected_population[i] = self.quantum_perturbation(selected_population[i], lb, ub)

                value = func(selected_population[i])
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = selected_population[i]

                if self.evaluations >= self.budget:
                    break

            population = selected_population

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def selection(self, population, func):
        fitness_values = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        sorted_indices = np.argsort(fitness_values)
        return population[sorted_indices[:self.population_size]]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def mutation(self, individual, lb, ub):
        mutation_vector = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
        return np.clip(individual + mutation_vector, lb, ub)

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def calculate_diversity(self, population):
        centroid = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - centroid, axis=1))
        return diversity