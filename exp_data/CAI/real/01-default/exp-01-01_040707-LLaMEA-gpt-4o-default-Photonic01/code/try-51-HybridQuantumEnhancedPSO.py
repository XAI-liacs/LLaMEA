import numpy as np

class HybridQuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.2
        self.w_max = 0.8
        self.rotation_angle = np.pi / 4
        self.mutation_rate = 0.1
        self.elite_archive_size = 5
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.elite_archive = []

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.elite_archive = []

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
            self.update_elite_archive(self.position[i], scores[i])
        return scores

    def update_elite_archive(self, position, score):
        if len(self.elite_archive) < self.elite_archive_size:
            self.elite_archive.append((position, score))
        else:
            worst_idx = max(range(len(self.elite_archive)), key=lambda i: self.elite_archive[i][1])
            if score < self.elite_archive[worst_idx][1]:
                self.elite_archive[worst_idx] = (position, score)

    def update_inertia_weight(self, iteration, max_iterations):
        return self.w_max - ((self.w_max - self.w_min) * np.log(iteration + 1) / np.log(max_iterations + 1))

    def quantum_rotation_gate(self, velocity):
        theta = np.random.rand(*velocity.shape) * self.rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def gaussian_mutation(self, position):
        mutation = np.random.randn(*position.shape) * self.mutation_rate
        return position + mutation

    def adaptive_feature_selection(self, position):
        selected_features = np.random.choice(self.dim, size=int(self.dim * 0.8), replace=False)
        new_position = np.copy(position)
        new_position[selected_features] += np.random.randn(len(selected_features)) * 0.05
        return new_position

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)

        quantum_velocity = self.quantum_rotation_gate(self.velocity)
        self.velocity = inertia_weight * quantum_velocity + cognitive + social
        self.position += self.velocity
        self.position = self.gaussian_mutation(self.position)

        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                self.position[i] = self.adaptive_feature_selection(self.position[i])

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(iteration, max_iterations)
            iteration += 1

        return self.gbest, self.gbest_score