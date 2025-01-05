import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
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
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_global_best(self):
        best_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[best_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()

    def quantum_entanglement(self):
        for i in range(self.population_size):
            entangled_state = np.mean(self.personal_best_positions, axis=0)
            self.velocities[i] += np.random.uniform(-1, 1, self.dim) * (entangled_state - self.population[i])
            self.velocities[i] = np.clip(self.velocities[i], -1, 1)

    def update_particle(self, i):
        inertia = self.inertia_weight * self.velocities[i]
        cognitive = self.cognitive_coefficient * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.population[i])
        social = self.social_coefficient * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
        self.velocities[i] = inertia + cognitive + social
        self.population[i] += self.velocities[i]
        self.population[i] = np.clip(self.population[i], self.func.bounds.lb, self.func.bounds.ub)

    def update_personal_best(self, i):
        score = self.evaluate(self.population[i])
        if score < self.personal_best_scores[i]:
            self.personal_best_scores[i] = score
            self.personal_best_positions[i] = self.population[i].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self.update_particle(i)
                self.update_personal_best(i)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.quantum_entanglement()
            self.update_global_best()

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}