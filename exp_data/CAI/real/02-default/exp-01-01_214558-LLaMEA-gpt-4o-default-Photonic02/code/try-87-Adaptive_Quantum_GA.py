import numpy as np

class Adaptive_Quantum_GA:
    def __init__(self, budget, dim, population_size=20, crossover_prob=0.7, mutation_prob=0.1, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.learning_rate = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_solution, best_value = None, float('inf')

        while self.evaluations < self.budget:
            new_population = []

            for i in range(0, self.population_size, 2):
                # Selection
                parent1, parent2 = self.tournament_selection(population, func), self.tournament_selection(population, func)

                # Crossover
                if np.random.rand() < self.crossover_prob:
                    offspring1, offspring2 = self.quantum_crossover(parent1, parent2, lb, ub)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Mutation
                if np.random.rand() < self.mutation_prob:
                    offspring1 = self.quantum_mutation(offspring1, lb, ub)
                    offspring2 = self.quantum_mutation(offspring2, lb, ub)

                if np.random.rand() < self.quantum_prob:
                    offspring1 = self.quantum_perturbation(offspring1, lb, ub)
                    offspring2 = self.quantum_perturbation(offspring2, lb, ub)

                new_population.extend([offspring1, offspring2])

            population = new_population

            # Evaluate new population
            for individual in population:
                value = func(individual)
                self.evaluations += 1

                if value < best_value:
                    best_value = value
                    best_solution = individual

                if self.evaluations >= self.budget:
                    break

        return best_solution

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def tournament_selection(self, population, func):
        tournament_size = 3
        selected = np.random.choice(population, tournament_size)
        best = min(selected, key=func)
        return best

    def quantum_crossover(self, parent1, parent2, lb, ub):
        alpha = np.random.rand(self.dim)
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        return np.clip(offspring1, lb, ub), np.clip(offspring2, lb, ub)

    def quantum_mutation(self, individual, lb, ub):
        mutation_vector = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(individual + mutation_vector, lb, ub)

    def quantum_perturbation(self, individual, lb, ub):
        perturbation = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(individual + perturbation, lb, ub)