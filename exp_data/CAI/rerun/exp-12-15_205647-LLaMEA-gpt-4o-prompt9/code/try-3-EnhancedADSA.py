import numpy as np

class EnhancedADSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, budget // 10)
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.729
        self.cognitive_coef = 1.494
        self.social_coef = 1.494

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        last_improvement = 0
        
        while eval_count < self.budget:
            # Evaluate current positions
            scores = np.array([func(x) for x in positions])
            eval_count += self.population_size
            
            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]
            
            # Update global best
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]
                last_improvement = eval_count
            
            # Adjust inertia weight based on stochastic factor
            if eval_count - last_improvement > self.population_size:
                self.inertia_weight = max(0.4, self.inertia_weight * (0.98 + 0.04 * np.random.rand()))
            
            # Dynamic adjustment of coefficients
            self.cognitive_coef = 1.494 + 0.5 * (1 - eval_count / self.budget)
            self.social_coef = 1.494 - 0.5 * (1 - eval_count / self.budget)
            
            # Dynamic resizing of population
            if eval_count > self.budget * 0.5:
                self.population_size = max(self.initial_population_size // 2, 10)
            
            # Update velocities and positions
            r1, r2 = np.random.uniform(size=(2, self.population_size, self.dim))
            cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coef * r2 * (global_best_position - positions)
            velocities = (self.inertia_weight * velocities + cognitive_velocity + social_velocity)
            
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_score