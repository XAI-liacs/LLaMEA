import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.4
        self.social_coefficient = 1.4
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        
    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Particle Swarm Optimization step
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coefficient * r2 * (global_best_position - positions[i]))
                candidates = positions[i] + velocities[i]
                
                # Differential Evolution step
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = positions[indices]
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate,
                                        mutant_vector, candidates)
                
                # Evaluate new candidate
                trial_score = func(trial_vector)
                evaluations += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector
            
            positions = np.array([personal_best_positions[i] if np.random.rand() < 0.5 else positions[i]
                                  for i in range(self.population_size)])
        
        return global_best_position