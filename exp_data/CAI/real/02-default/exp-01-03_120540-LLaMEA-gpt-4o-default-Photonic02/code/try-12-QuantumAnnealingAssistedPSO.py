import numpy as np

class QuantumAnnealingAssistedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
    
    def quantum_annealing_update(self, position, global_best):
        q_anneal_factor = np.random.normal(0, 1, self.dim)
        new_position = position + q_anneal_factor * (global_best - position)
        return new_position
    
    def particle_swarm_optimization(self, func, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_coefficient * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
            
            score = func(self.positions[i])
            evaluations += 1
            
            if score < self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i]
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]
        
        return evaluations
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            evaluations = self.particle_swarm_optimization(func, evaluations)
            
            # Quantum annealing inspired global update
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                new_position = self.quantum_annealing_update(self.positions[i], self.global_best_position)
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                new_score = func(new_position)
                evaluations += 1
                
                if new_score < self.best_scores[i]:
                    self.best_scores[i] = new_score
                    self.best_positions[i] = new_position
                
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best_position = new_position