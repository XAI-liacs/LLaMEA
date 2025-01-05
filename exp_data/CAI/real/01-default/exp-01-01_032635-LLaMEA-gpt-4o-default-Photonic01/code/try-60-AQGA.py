import numpy as np

class AQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.fitness = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.exploration_boost = 0.05
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1, self.dim)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return self.apply_quantum_boost(offspring)

    def apply_quantum_boost(self, position):
        boost = np.random.normal(0, 1, self.dim) * self.exploration_boost
        new_position = position + boost
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def mutate(self, position):
        mutation_vector = np.random.uniform(-self.mutation_rate, self.mutation_rate, self.dim)
        return np.clip(position + mutation_vector, self.bounds[0], self.bounds[1])

    def select_parents(self):
        # Tournament selection
        candidates = np.random.choice(self.population_size, 2, replace=False)
        if self.fitness[candidates[0]] < self.fitness[candidates[1]]:
            return self.population[candidates[0]], self.population[candidates[1]]
        else:
            return self.population[candidates[1]], self.population[candidates[0]]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.population[i])
                evaluations += 1

                if current_value < self.fitness[i]:
                    self.fitness[i] = current_value

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.population[i].copy()

            # Create next generation
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                if np.random.rand() < self.crossover_rate:
                    offspring1 = self.quantum_crossover(parent1, parent2)
                    offspring2 = self.quantum_crossover(parent2, parent1)
                else:
                    offspring1, offspring2 = parent1, parent2

                if np.random.rand() < self.mutation_rate:
                    offspring1 = self.mutate(offspring1)
                if np.random.rand() < self.mutation_rate:
                    offspring2 = self.mutate(offspring2)

                new_population.extend([offspring1, offspring2])

            self.population = np.array(new_population)

        return self.global_best_position, self.global_best_value