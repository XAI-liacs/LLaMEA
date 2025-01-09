import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // 2)
        self.velocity_clamp = (self.lower_bound, self.upper_bound)
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        global_best_position = None
        global_best_score = float('inf')
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Evaluate current solution
                score = func(population[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
            
            # Update inertia weight dynamically
            self.inertia_weight = 0.4 + (0.3 * (self.budget - evaluations) / self.budget)
            
            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive_component = self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                social_component = self.social_weight * r2 * (global_best_position - population[i])
                
                velocities[i] = (self.inertia_weight * velocities[i]) + cognitive_component + social_component
                velocities[i] = np.clip(velocities[i], *self.velocity_clamp)
                
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
            
            # Introduce mutation to avoid local optima stagnation
            mutation_probability = 0.1
            for i in range(self.population_size):
                if np.random.rand() < mutation_probability:
                    mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    population[i] = mutation_vector
        
        return global_best_position