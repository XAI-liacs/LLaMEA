import numpy as np

class AdaptiveSwarmEvolutionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Typical swarm size
        self.inertia_weight = 0.7  # Increased inertia for better exploration
        self.cognitive_component = 1.4  # Reduced cognitive component for balance
        self.social_component = 1.8  # Modified social component for stronger convergence
        self.adaptive_mutation_rate = 0.05  # Reduced adaptive mutation rate
        self.mutation_scale = 0.12  # Further refined mutation scale for exploration

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        
        while eval_count < self.budget:
            scores = np.array([func(pos) for pos in positions])
            eval_count += self.population_size
            
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]
            
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_component * r1 * (personal_best_positions - positions)
                          + self.social_component * r2 * (global_best_position - positions))
            positions += velocities
            
            mutation_rate = self.adaptive_mutation_rate * (1 - eval_count / self.budget)
            dynamic_mutation_scale = self.mutation_scale * (1 + eval_count / self.budget)
            mutations = np.random.uniform(-1, 1, (self.population_size, self.dim))
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate
            positions += mutation_mask * mutations * dynamic_mutation_scale

            positions = np.clip(positions, lb, ub)

        return global_best_position, global_best_score