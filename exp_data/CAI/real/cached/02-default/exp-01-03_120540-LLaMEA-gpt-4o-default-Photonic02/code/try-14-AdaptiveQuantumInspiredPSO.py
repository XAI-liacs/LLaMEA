import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = np.zeros((self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.zeros((self.population_size, self.dim))
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.9   # Inertia weight
        
    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        
    def adaptive_learning_coefficients(self, evaluations):
        t = evaluations / self.budget
        self.c1 = 2.5 - 1.5 * t
        self.c2 = 0.5 + 1.5 * t
        self.w = 0.9 - 0.7 * t
        
    def quantum_inspired_update(self):
        quantum_bit = np.random.rand(self.population_size, self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        self.positions += quantum_flip
        
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_score = func(self.positions[i])
                evaluations += 1
                
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]
                
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]
        
                # Update velocities and positions
                self.adaptive_learning_coefficients(evaluations)
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Bound handling
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                
            # Quantum-inspired global update
            self.quantum_inspired_update()
            self.positions = np.clip(self.positions, func.bounds.lb, func.bounds.ub)
        
        return self.global_best_position, self.global_best_score