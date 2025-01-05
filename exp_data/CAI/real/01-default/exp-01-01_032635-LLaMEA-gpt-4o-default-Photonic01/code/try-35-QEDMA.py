import numpy as np

class QEDMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.individuals = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.meme_prob = 0.2
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.individuals = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.personal_best_positions = self.individuals.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_inspired_exploration(self, individual, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        new_individual = individual + beta * (global_best - individual) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(new_individual, lb, ub)

    def adaptive_meme_search(self, individual):
        if np.random.rand() < self.meme_prob:
            perturbation = (np.random.rand(self.dim) - 0.5) * 0.1
            individual += perturbation
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = (np.random.rand(self.dim) - 0.5) * 0.1
            return np.clip(individual + mutation_vector, self.bounds[0], self.bounds[1])
        return individual

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.individuals[i])
                evaluations += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.individuals[i].copy()

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.individuals[i].copy()

            new_population = []

            for j in range(0, self.population_size, 2):
                parent1, parent2 = self.individuals[np.random.choice(self.population_size, 2, replace=False)]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.append(child1)
                new_population.append(child2)

            self.individuals = np.array(new_population[:self.population_size])

            for k in range(self.population_size):
                self.individuals[k] = self.quantum_inspired_exploration(self.individuals[k], self.global_best_position)
                self.individuals[k] = self.adaptive_meme_search(self.individuals[k])

        return self.global_best_position, self.global_best_value