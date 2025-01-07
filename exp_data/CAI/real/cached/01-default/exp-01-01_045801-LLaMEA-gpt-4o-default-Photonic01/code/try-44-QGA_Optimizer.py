import numpy as np

class QGA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.evaluations = 0
        self.mutation_rate = 0.1

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_superposition(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        return np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

    def mutate(self, individual):
        mutation_vector = np.random.uniform(-1, 1, self.dim)
        mutated_individual = individual + self.mutation_rate * mutation_vector
        return np.clip(mutated_individual, self.func.bounds.lb, self.func.bounds.ub)

    def select_parents(self):
        indices = np.random.choice(self.population_size, 2, replace=False)
        return self.population[indices[0]], self.population[indices[1]]

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            
            self.population = np.array(new_population)
            self.scores = np.array([self.evaluate(ind) for ind in self.population])
            self.update_best()

        return {'solution': self.best_solution, 'fitness': self.best_score}