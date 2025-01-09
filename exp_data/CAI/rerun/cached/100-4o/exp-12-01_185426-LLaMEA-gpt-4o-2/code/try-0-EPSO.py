import numpy as np

class EPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.mutation_rate = 0.1
    
    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, 
                                      (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate fitness
            scores = np.array([func(pos) for pos in positions])
            evaluations += self.population_size
            
            # Update personal bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    
            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best_position = positions[min_score_idx]

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_const * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_const * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity + social_velocity)
                positions[i] += velocities[i]
                
                # Apply boundaries
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
                
                # Mutation strategy to escape local optima
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    positions[i] += mutation_vector
                    positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
        
        return global_best_position