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
    
    def adaptive_quantum_mutation(self, individual, evaluations):
        # Adaptive quantum mutation based on remaining budget
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutation_intensity = self.mutation_factor * (1 - evaluations / self.budget)
        mutated_individual = individual + mutation_intensity * quantum_flip
        return mutated_individual
    
    def differential_evolution(self, func, target_idx):
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
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.differential_evolution(func, i)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                # Adaptive mutation factor based on progress
                self.mutation_factor = self.initial_mutation_factor * (1 - evaluations / self.budget)
                
                # Dynamic population size adjustment
                if evaluations % (self.budget // 10) == 0 and self.population_size > 10:
                    self.population_size -= 1
                    self.positions = self.positions[:self.population_size]

        # Quantum-inspired global search with adaptive mutation
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            mutated_position = self.adaptive_quantum_mutation(self.positions[i], evaluations)
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position