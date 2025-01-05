import numpy as np

class QE_APSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1_min = 1.5
        self.c1_max = 2.5
        self.c2_min = 1.5
        self.c2_max = 2.5
        self.population = None
        self.velocities = None
        self.scores = None
        self.best_individual_positions = None
        self.best_individual_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.best_individual_positions = self.population.copy()
        self.best_individual_scores = self.scores.copy()
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_global_best(self):
        min_idx = np.argmin(self.scores)
        if self.scores[min_idx] < self.global_best_score:
            self.global_best_score = self.scores[min_idx]
            self.global_best_position = self.population[min_idx].copy()

    def adapt_parameters(self, iteration_ratio):
        w = self.w_max - (self.w_max - self.w_min) * iteration_ratio
        c1 = self.c1_min + (self.c1_max - self.c1_min) * iteration_ratio
        c2 = self.c2_min + (self.c2_max - self.c2_min) * iteration_ratio
        return w, c1, c2

    def quantum_position_update(self):
        quantum_superposition = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return self.global_best_position + quantum_superposition * np.abs(self.population - self.global_best_position)

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        max_iterations = self.budget // self.population_size

        for iter_num in range(max_iterations):
            iteration_ratio = iter_num / max_iterations
            w, c1, c2 = self.adapt_parameters(iteration_ratio)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = c1 * r1 * (self.best_individual_positions[i] - self.population[i])
                social = c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                quantum_update = self.quantum_position_update()[i]
                self.population[i] = np.clip(self.population[i] + self.velocities[i] + quantum_update, lb, ub)

                self.scores[i] = self.evaluate(self.population[i])
                self.evaluations += 1

                if self.scores[i] < self.best_individual_scores[i]:
                    self.best_individual_scores[i] = self.scores[i]
                    self.best_individual_positions[i] = self.population[i].copy()

            self.update_global_best()

            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}