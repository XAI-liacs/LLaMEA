import numpy as np

class BioInspiredSwarmWithAdaptiveChaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.chaos_factor = 0.1
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
    
    def adaptive_chaos_search(self, individual):
        chaos_perturbation = (np.random.rand(self.dim) - 0.5) * 2
        chaotic_individual = individual + self.chaos_factor * chaos_perturbation
        return chaotic_individual
    
    def update_velocities_and_positions(self, func):
        w = 0.5  # inertia weight
        c1, c2 = 1.5, 1.5  # acceleration coefficients
        
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = c2 * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
    
    def evaluate_swarm(self, func):
        for i in range(self.swarm_size):
            score = func(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = np.copy(self.positions[i])
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = np.copy(self.positions[i])
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.update_velocities_and_positions(func)
            self.evaluate_swarm(func)
            evaluations += self.swarm_size
            
            # Adaptive chaotic search for enhanced exploration
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                chaotic_position = self.adaptive_chaos_search(self.positions[i])
                chaotic_position = np.clip(chaotic_position, func.bounds.lb, func.bounds.ub)
                chaotic_score = func(chaotic_position)
                evaluations += 1
                
                if chaotic_score < self.global_best_score:
                    self.global_best_score = chaotic_score
                    self.global_best_position = chaotic_position
            
            # Adjust chaos factor based on progress
            self.chaos_factor *= (1 - evaluations / self.budget)