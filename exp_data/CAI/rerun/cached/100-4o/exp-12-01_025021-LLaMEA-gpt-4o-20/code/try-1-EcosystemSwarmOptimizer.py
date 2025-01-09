import numpy as np

class EcosystemSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.iterations = self.budget // self.population_size
    
    def __call__(self, func):
        eval_count = 0
        for _ in range(self.iterations):
            for i in range(self.population_size):
                score = func(self.population[i])
                eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.population[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.population[i].copy()
            
            for i in range(self.population_size):
                local_best = self._find_local_best(i)
                inertia_weight = 0.5 + 0.5 * (self.global_best_score / (1e-8 + np.min(self.personal_best_scores)))  # Adaptive inertia
                cognitive_component = 1.5 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])  # Reduced cognitive influence
                social_component = 2.5 * np.random.rand(self.dim) * (local_best - self.population[i])  # Increased social influence
                
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.population[i] += self.velocities[i]
                
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
            
            if eval_count >= self.budget:
                break
        
        return self.global_best, self.global_best_score

    def _find_local_best(self, index):
        neighborhood_size = max(1, self.population_size // 5)  # Larger neighborhood
        neighbors = np.random.choice(self.population_size, neighborhood_size, replace=False)
        local_best = self.population[neighbors[0]]
        local_best_score = self.personal_best_scores[neighbors[0]]
        
        for neighbor in neighbors:
            if self.personal_best_scores[neighbor] < local_best_score:
                local_best_score = self.personal_best_scores[neighbor]
                local_best = self.population[neighbor]
        
        return local_best