import numpy as np

class AdaptivePredatorPreyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.predator_ratio = 0.2
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.learning_rate = 0.1
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def predator_prey_interaction(self, func, evaluations):
        predators_count = int(self.population_size * self.predator_ratio)
        predators_idx = np.random.choice(self.population_size, predators_count, replace=False)
        prey_idx = [i for i in range(self.population_size) if i not in predators_idx]
        
        for predator in predators_idx:
            if evaluations >= self.budget:
                break
            for prey in prey_idx:
                if evaluations >= self.budget:
                    break
                
                # Predator hunts prey
                movement_vector = self.positions[predator] - self.positions[prey]
                self.positions[prey] += self.learning_rate * movement_vector
                self.positions[prey] = np.clip(self.positions[prey], func.bounds.lb, func.bounds.ub)
                
                prey_score = func(self.positions[prey])
                evaluations += 1
                
                if prey_score < self.best_score:
                    self.best_score = prey_score
                    self.best_position = self.positions[prey]

                    # Swap roles if prey is stronger (better score)
                    self.positions[predator], self.positions[prey] = self.positions[prey], self.positions[predator]
    
    def adaptive_learning(self, individual, bounds, evaluations):
        learning_step = self.learning_rate * (1 - evaluations / self.budget)
        adjustment = np.random.normal(0, learning_step, self.dim)
        individual += adjustment
        individual = np.clip(individual, bounds.lb, bounds.ub)
        return individual
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.predator_prey_interaction(func, evaluations)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Adaptive learning phase
                new_position = self.adaptive_learning(self.positions[i], func.bounds, evaluations)
                new_score = func(new_position)
                evaluations += 1
                
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_position = new_position
                
                # Update if better
                if new_score < func(self.positions[i]):
                    self.positions[i] = new_position