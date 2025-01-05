import numpy as np

class EnhancedQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.initial_mutation_factor = 0.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_inspired_mutation(self, individual):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_individual = individual + self.mutation_factor * quantum_flip
        return mutated_individual
    
    def differential_evolution(self, func, target_idx, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        trial_vector = np.copy(self.positions[target_idx])
        
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        
        trial_vector = np.clip(trial_vector, lb, ub)
        return trial_vector, func(trial_vector)
    
    def dynamic_population_adjustment(self, evaluations):
        # Adjust population size dynamically based on optimization progress
        reduction_factor = 1 - (evaluations / self.budget)
        new_population_size = max(5, int(self.initial_population_size * reduction_factor))
        if new_population_size < self.population_size:
            self.population_size = new_population_size
            self.positions = self.positions[:self.population_size]

    def dimension_reduction(self):
        # Reduce dimensionality by focusing on promising dimensions
        # Select dimensions with high variance and drop those with low variance
        variance_threshold = 0.1
        variances = np.var(self.positions, axis=0)
        selected_dims = variances >= variance_threshold
        if np.any(selected_dims):
            self.positions = self.positions[:, selected_dims]
            self.dim = np.sum(selected_dims)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.differential_evolution(func, i, evaluations)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                self.mutation_factor = self.initial_mutation_factor * (1 - evaluations / self.budget)
            
            self.dynamic_population_adjustment(evaluations)
            self.dimension_reduction()
        
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            mutated_position = self.quantum_inspired_mutation(self.positions[i])
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position