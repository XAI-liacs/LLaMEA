import numpy as np

class QuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_scores = None
        self.gbest_position = None
        self.gbest_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.population_size, float('inf'))
    
    def quantum_superposition(self, position):
        # Introduce quantum superposition to enhance exploration
        superposition = np.random.rand(self.dim) * (np.random.choice([-1, 1], self.dim))
        return position + superposition
    
    def update_velocity_and_position(self, func, idx, w):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = self.c1 * r1 * (self.pbest_positions[idx] - self.positions[idx])
        social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[idx])
        new_velocity = w * self.velocities[idx] + cognitive_velocity + social_velocity
        new_position = self.positions[idx] + new_velocity
        
        # Apply quantum superposition to the new position
        quantum_position = self.quantum_superposition(new_position)
        quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
        
        return new_velocity, quantum_position, func(quantum_position)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                new_velocity, new_position, new_score = self.update_velocity_and_position(func, i, w)
                evaluations += 1
                
                if new_score < self.pbest_scores[i]:
                    self.pbest_scores[i] = new_score
                    self.pbest_positions[i] = new_position
                
                if new_score < self.gbest_score:
                    self.gbest_score = new_score
                    self.gbest_position = new_position
                
                self.velocities[i] = new_velocity
                self.positions[i] = new_position