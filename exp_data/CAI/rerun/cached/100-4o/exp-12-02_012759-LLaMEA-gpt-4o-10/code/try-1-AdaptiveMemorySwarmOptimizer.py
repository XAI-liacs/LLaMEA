import numpy as np

class AdaptiveMemorySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.memory = []
    
    def evaluate_population(self, func, population):
        return np.array([func(x) for x in population])
    
    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = self.evaluate_population(func, personal_best_positions)
        
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.population_size
        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - population) +
                self.social_coeff * r2 * (global_best_position - population)
            )
            population += velocities
            population = np.clip(population, self.lower_bound, self.upper_bound)
            
            scores = self.evaluate_population(func, population)
            evaluations += self.population_size
            
            improved = scores < personal_best_scores
            personal_best_positions[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            
            if np.min(scores) < global_best_score:
                global_best_index = np.argmin(scores)
                global_best_position = population[global_best_index]
                global_best_score = scores[global_best_index]
            
            self.memory.append(global_best_position)
            if len(self.memory) > 10:
                self.memory.pop(0)
            
            # Adaptive mechanism: adjust inertia weight based on improvement
            if len(self.memory) > 1 and np.var([func(pos) for pos in self.memory]) < 1e-4:
                self.inertia_weight *= 0.9
            else:
                self.inertia_weight = min(0.9, self.inertia_weight * 1.1)
        
        return global_best_position, global_best_score