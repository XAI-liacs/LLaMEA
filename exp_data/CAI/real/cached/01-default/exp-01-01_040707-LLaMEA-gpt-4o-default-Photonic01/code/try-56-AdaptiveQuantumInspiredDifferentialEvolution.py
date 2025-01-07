import numpy as np

class AdaptiveQuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.rotation_angle = np.pi / 4
        self.adapt_factor = 0.1
        self.position = None
        self.best_position = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_position = np.copy(self.position[0])

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_position = self.position[i]
        return scores

    def quantum_rotation(self, vector):
        theta = np.random.rand(*vector.shape) * self.rotation_angle
        q_vector = vector * np.cos(theta) + np.random.rand(*vector.shape) * np.sin(theta)
        return q_vector

    def differential_evolution(self, func):
        for i in range(self.population_size):
            indices = list(range(0, i)) + list(range(i + 1, self.population_size))
            a, b, c = self.position[np.random.choice(indices, 3, replace=False)]
            mutant_vector = self.quantum_rotation(a + self.mutation_factor * (b - c))
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            trial_vector = np.where(cross_points, mutant_vector, self.position[i])
            trial_score = func(trial_vector)
            if trial_score < func(self.position[i]):
                self.position[i] = trial_vector

    def update_parameters(self, iteration, max_iterations):
        self.mutation_factor = 0.5 + (0.5 - 0.1) * (1 - iteration / max_iterations) ** 2
        self.crossover_prob = 0.9 - 0.4 * np.sin(np.pi * iteration / max_iterations)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            self.evaluate(func)
            func_calls += self.population_size
            self.differential_evolution(func)
            self.update_parameters(iteration, max_iterations)
            iteration += 1

        return self.best_position, self.best_score