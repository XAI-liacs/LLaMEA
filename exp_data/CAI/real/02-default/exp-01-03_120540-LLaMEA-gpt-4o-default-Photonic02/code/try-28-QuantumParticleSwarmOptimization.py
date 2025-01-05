import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5

    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.copy(self.positions[0])

    def update_velocities_and_positions(self, func):
        r1, r2 = np.random.rand(2, self.population_size, self.dim)
        for i in range(self.population_size):
            cognitive_velocity = self.cognitive_component * r1[i] * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social_component * r2[i] * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

    def quantum_position_update(self):
        for i in range(self.population_size):
            random_pos = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            offset = np.random.rand(self.dim) < 0.5
            self.positions[i] = offset * self.positions[i] + (1 - offset) * random_pos

    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
                    
            self.update_velocities_and_positions(func)
        
        # Perform a quantum position update to exploit and explore
        self.quantum_position_update()
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break

            score = func(self.positions[i])
            evaluations += 1
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]