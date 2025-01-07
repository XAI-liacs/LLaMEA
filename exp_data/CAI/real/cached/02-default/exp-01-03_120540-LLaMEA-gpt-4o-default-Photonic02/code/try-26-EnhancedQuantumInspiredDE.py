import numpy as np

class EnhancedQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.initial_mutation_factor = 0.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def adaptive_quantum_inspired_perturbation(self, individual):
        # Enhanced adaptive quantum-inspired perturbation
        perturbation_strength = self.mutation_factor * np.exp(-self.best_score)
        quantum_bit = np.random.rand(self.dim) < perturbation_strength
        quantum_flip = np.where(quantum_bit, 1, -1)
        perturbed_individual = individual + perturbation_strength * quantum_flip
        return perturbed_individual
    
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
                
                # Adaptive mutation factor based on progress
                self.mutation_factor = self.initial_mutation_factor * (1 - evaluations / self.budget)
        
        # Quantum-inspired global search with enhanced perturbation
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            perturbed_position = self.adaptive_quantum_inspired_perturbation(self.positions[i])
            perturbed_position = np.clip(perturbed_position, func.bounds.lb, func.bounds.ub)
            perturbed_score = func(perturbed_position)
            evaluations += 1
            
            if perturbed_score < self.best_score:
                self.best_score = perturbed_score
                self.best_position = perturbed_position