import numpy as np

class AdaptiveSwarmEvolutionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Typical swarm size
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.9  # Improved social component for stronger information sharing
        self.adaptive_mutation_rate = 0.1
        self.mutation_scale = 0.16  # Changed mutation scale for enhanced exploration

    def __call__(self, func):
        # Initialize swarm positions and velocities
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        
        while eval_count < self.budget:
            # Evaluate function
            scores = np.array([func(pos) for pos in positions])
            eval_count += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]
            
            # Update velocities and positions
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_component * r1 * (personal_best_positions - positions)
                          + self.social_component * r2 * (global_best_position - positions))
            positions += velocities
            
            # Adaptive mutation
            mutation_rate = self.adaptive_mutation_rate * (1 - eval_count / self.budget)
            mutations = np.random.uniform(-1, 1, (self.population_size, self.dim))
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate
            positions += mutation_mask * mutations * self.mutation_scale
            
            # Ensure positions are within bounds
            positions = np.clip(positions, lb, ub)
        
        return global_best_position, global_best_score