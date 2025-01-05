import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.population = None
        self.velocities = None
        self.p_best_positions = None
        self.p_best_scores = None
        self.g_best_position = None
        self.g_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        self.p_best_positions = self.population.copy()
        self.p_best_scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_personal_best(self, idx):
        score = self.evaluate(self.population[idx])
        if score < self.p_best_scores[idx]:
            self.p_best_scores[idx] = score
            self.p_best_positions[idx] = self.population[idx].copy()
        return score

    def update_global_best(self):
        min_idx = np.argmin(self.p_best_scores)
        if self.p_best_scores[min_idx] < self.g_best_score:
            self.g_best_score = self.p_best_scores[min_idx]
            self.g_best_position = self.p_best_positions[min_idx].copy()

    def quantum_tunneling(self):
        for i in range(self.population_size):
            quantum_factor = np.random.uniform(-0.1, 0.1, self.dim)
            quantum_position = self.g_best_position + quantum_factor * np.abs(self.population[i] - self.g_best_position)
            quantum_position = np.clip(quantum_position, self.func.bounds.lb, self.func.bounds.ub)
            quantum_score = self.evaluate(quantum_position)
            if quantum_score < self.p_best_scores[i]:
                self.population[i] = quantum_position
                self.p_best_scores[i] = quantum_score
                self.p_best_positions[i] = quantum_position
                if quantum_score < self.g_best_score:
                    self.g_best_score = quantum_score
                    self.g_best_position = quantum_position.copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.p_best_positions[i] - self.population[i])
                social_velocity = self.c2 * r2 * (self.g_best_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], lb, ub)
                self.update_personal_best(i)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.update_global_best()
            self.quantum_tunneling()

        return {'solution': self.g_best_position, 'fitness': self.g_best_score}