import numpy as np

class EnhancedQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.mutation_factor = self.initial_mutation_factor
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_inspired_mutation(self, individual, diversity_factor):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, diversity_factor, -diversity_factor)
        mutated_individual = individual + self.mutation_factor * quantum_flip
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
    
    def entropy_based_adaptive_diversity(self):
        population_entropy = np.mean(np.std(self.positions, axis=0))
        diversity_factor = min(1, max(0.1, 1 - population_entropy))
        return diversity_factor
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                diversity_factor = self.entropy_based_adaptive_diversity()
                trial_vector, trial_score = self.differential_evolution(func, i)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                self.mutation_factor = self.initial_mutation_factor * (1 - evaluations / self.budget)
            
            # Dynamically adjust population size
            if evaluations > self.budget * 0.5 and self.population_size > 10:
                self.population_size -= 1
                self.positions = self.positions[:self.population_size]
        
        # Final quantum-inspired exploration
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            diversity_factor = self.entropy_based_adaptive_diversity()
            mutated_position = self.quantum_inspired_mutation(self.positions[i], diversity_factor)
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position