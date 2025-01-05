import numpy as np

class QPF_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.global_best = None
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.alpha = 0.1
        self.beta = 2.0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, solution):
        return self.func(solution)

    def update_best(self):
        for i in range(self.population_size):
            score = self.evaluate(self.population[i])
            if score < self.scores[i]:
                self.scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.population[i].copy()

    def quantum_particle_fusion(self, lb, ub):
        quantum_states = np.random.uniform(-1, 1, (self.population_size, self.dim))
        for i in range(self.population_size):
            fusion_vector = self.global_best + self.alpha * quantum_states[i] * np.abs(self.population[i] - self.global_best)
            fusion_vector = np.clip(fusion_vector, lb, ub)
            score = self.evaluate(fusion_vector)
            if score < self.scores[i]:
                self.population[i] = fusion_vector
                self.scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = fusion_vector.copy()

    def update_population(self, lb, ub):
        for i in range(self.population_size):
            rand_idx = np.random.randint(self.population_size)
            selected_particle = self.population[rand_idx]
            velocity = self.beta * (selected_particle - self.population[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
            new_position = self.population[i] + velocity
            new_position = np.clip(new_position, lb, ub)
            score = self.evaluate(new_position)
            if score < self.scores[i]:
                self.population[i] = new_position
                self.scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = new_position.copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)
        self.update_best()

        while self.evaluations < self.budget:
            self.update_population(lb, ub)
            self.evaluations += self.population_size
            self.quantum_particle_fusion(lb, ub)
            self.evaluations += self.population_size

        return {'solution': self.global_best, 'fitness': self.global_best_score}