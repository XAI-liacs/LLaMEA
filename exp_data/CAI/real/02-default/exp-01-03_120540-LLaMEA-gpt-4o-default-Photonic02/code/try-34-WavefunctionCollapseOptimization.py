import numpy as np

class WavefunctionCollapseOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.entropy_weights = np.ones(dim)
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def collapse_wavefunction(self, individual, weights):
        # Wavefunction collapse inspired perturbation
        probabilities = weights / weights.sum()
        perturbation = np.random.choice([-1, 1], size=self.dim, p=[0.5, 0.5])
        collapsed_position = individual + probabilities * perturbation
        return collapsed_position
    
    def evolutionary_step(self, func, target_idx):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        # Differential evolution inspired mutation
        mutant_vector = self.positions[a] + 0.8 * (self.positions[b] - self.positions[c])
        trial_vector = np.copy(self.positions[target_idx])
        
        crossover_mask = np.random.rand(self.dim) < 0.9
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        
        trial_vector = np.clip(trial_vector, lb, ub)
        trial_score = func(trial_vector)
        return trial_vector, trial_score
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.evolutionary_step(func, i)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector

                    # Adjust entropy weights based on improvement
                    self.entropy_weights = np.maximum(self.entropy_weights * 0.9, 0.1)
                else:
                    self.entropy_weights = np.minimum(self.entropy_weights * 1.1, 1.0)
        
        # Final wavefunction collapse inspired exploration
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            collapsed_position = self.collapse_wavefunction(self.positions[i], self.entropy_weights)
            collapsed_position = np.clip(collapsed_position, func.bounds.lb, func.bounds.ub)
            collapsed_score = func(collapsed_position)
            evaluations += 1
            
            if collapsed_score < self.best_score:
                self.best_score = collapsed_score
                self.best_position = collapsed_position