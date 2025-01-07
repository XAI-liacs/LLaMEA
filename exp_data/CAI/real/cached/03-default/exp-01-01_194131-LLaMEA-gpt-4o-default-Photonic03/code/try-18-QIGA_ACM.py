import numpy as np

class QIGA_ACM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1

    def adaptive_mutation(self, individual):
        mutation_strength = np.random.rand(self.dim) * (self.best_score / (self.scores.min() + 1e-10))
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        return np.clip(individual + mutation_vector, 0, 1)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            alpha = np.random.rand(self.dim)
            child = alpha * parent1 + (1 - alpha) * parent2
        else:
            child = parent1
        return np.clip(child, 0, 1)

    def select_parents(self):
        idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
        return self.population[idx1], self.population[idx2]

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.population[i])
            self.scores[i] = score
            if score < self.best_score:
                self.best_solution = self.population[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                if np.random.rand() < self.mutation_prob:
                    child = self.adaptive_mutation(child)
                child_score = func(child)
                if child_score < self.best_score:
                    self.best_solution = child
                    self.best_score = child_score
                if child_score < self.scores[i]:
                    self.population[i] = child
                    self.scores[i] = child_score
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return self.best_solution