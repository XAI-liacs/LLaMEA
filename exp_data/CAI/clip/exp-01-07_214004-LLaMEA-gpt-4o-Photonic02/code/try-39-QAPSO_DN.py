import numpy as np

class QAPSO_DN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.quantum_component = 0.5
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        self.init_population(func.bounds.lb, func.bounds.ub)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            self.update_velocities_and_positions(func.bounds.lb, func.bounds.ub)
            self.adapt_parameters()
        return self.global_best_position, self.global_best_score

    def init_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))

    def update_velocities_and_positions(self, lb, ub):
        for i in range(self.population_size):
            local_neighbors = self.get_dynamic_neighborhood(i)
            local_best_position = min(local_neighbors, key=lambda x: self.personal_best_scores[x])
            
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            r3 = np.random.random(self.dim)

            cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social_component * r2 * (self.personal_best_positions[local_best_position] - self.positions[i])
            quantum_velocity = self.quantum_component * r3 * (self.global_best_position - self.positions[i])

            self.velocities[i] = self.inertia * self.velocities[i] + cognitive_velocity + social_velocity + quantum_velocity
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)

    def get_dynamic_neighborhood(self, index):
        neighborhood_size = max(5, self.population_size // 6)  # Changed neighborhood size
        distances = np.linalg.norm(self.positions - self.positions[index], axis=1)
        neighbors = np.argsort(distances)
        return neighbors[:neighborhood_size]

    def adapt_parameters(self):
        if self.evaluations > self.budget // 2:
            self.inertia = 0.5 + 0.1 * np.sin(self.evaluations / float(self.budget) * np.pi)
            self.cognitive_component *= 0.99
            self.social_component *= 1.01
            self.quantum_component *= 0.995  # Adjusted decay factor for quantum component