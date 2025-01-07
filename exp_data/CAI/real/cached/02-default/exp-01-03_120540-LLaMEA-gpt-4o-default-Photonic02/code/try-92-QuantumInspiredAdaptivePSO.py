import numpy as np

class QuantumInspiredAdaptivePSO:
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
        self.learning_factor = 0.5
        self.inertia_weight = 0.7
        self.initial_inertia_weight = 0.7
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
    
    def update_velocities_and_positions(self, func, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = self.learning_factor * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.learning_factor * r2 * (self.best_position - self.positions[i])
            quantum_component = self.quantum_component()

            self.velocities[i] = (self.inertia_weight * self.velocities[i] 
                                  + cognitive_component + social_component + quantum_component)
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)
    
    def quantum_component(self):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        return self.learning_factor * quantum_flip

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i].copy()
            
            self.update_velocities_and_positions(func, evaluations)
            
            # Adaptive inertia weight based on progress
            self.inertia_weight = self.initial_inertia_weight * (1 - evaluations / self.budget)