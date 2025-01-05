import numpy as np

class AdaptiveQuantumEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.rotation_angle = np.pi / 4
        self.position = None
        self.best_solutions = None
        self.best_scores = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_solutions = np.copy(self.position)
        self.best_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_solutions[i] = self.position[i]
        return scores

    def quantum_rotation_gate(self, position):
        theta = np.random.rand(*position.shape) * self.rotation_angle
        q_position = position * np.cos(theta) + np.random.rand(*position.shape) * np.sin(theta)
        return q_position

    def mutate(self, position):
        mutation = np.random.randn(*position.shape) * self.mutation_rate
        return position + mutation

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        else:
            child = parent1
        return child

    def select_parents(self, scores):
        probabilities = 1 / (scores + 1e-9)
        probabilities /= probabilities.sum()
        parents_indices = np.random.choice(self.population_size, size=2, p=probabilities)
        return self.position[parents_indices[0]], self.position[parents_indices[1]]

    def adaptive_elitism(self, scores):
        elite_size = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(scores)[:elite_size]
        return self.position[elite_indices]

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        generation = 0
        max_generations = self.budget // self.population_size

        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size

            elite_individuals = self.adaptive_elitism(scores)
            next_generation = []

            while len(next_generation) < self.population_size:
                parent1, parent2 = self.select_parents(scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            self.position = np.array(next_generation)
            self.position[:len(elite_individuals)] = elite_individuals  # Preserve elites
            self.position = self.quantum_rotation_gate(self.position)
            generation += 1

        best_index = np.argmin(self.best_scores)
        return self.best_solutions[best_index], self.best_scores[best_index]