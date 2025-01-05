import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        for i in range(self.population_size):
            score = self.evaluate(self.population[i])
            self.update_personal_best(i, score)

    def evaluate(self, solution):
        return self.func(solution)
    
    def update_personal_best(self, idx, score):
        if score < self.personal_best_scores[idx]:
            self.personal_best_scores[idx] = score
            self.personal_best_positions[idx] = self.population[idx].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[idx].copy()

    def update_velocities_and_positions(self, lb, ub):
        w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.population[i])
            self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
            self.population[i] = self.population[i] + self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)
    
    def quantum_boundary_exploration(self, lb, ub):
        quantum_factor = np.random.uniform(-1, 1, (self.population_size, self.dim))
        for i in range(self.population_size):
            candidate = self.global_best_position + quantum_factor[i] * (self.population[i] - self.global_best_position)
            candidate = np.clip(candidate, lb, ub)
            score = self.evaluate(candidate)
            self.evaluations += 1
            self.update_personal_best(i, score)
            if self.evaluations >= self.budget:
                break

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_velocities_and_positions(lb, ub)
            for i in range(self.population_size):
                score = self.evaluate(self.population[i])
                self.evaluations += 1
                self.update_personal_best(i, score)
                if self.evaluations >= self.budget:
                    break
            self.quantum_boundary_exploration(lb, ub)

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}