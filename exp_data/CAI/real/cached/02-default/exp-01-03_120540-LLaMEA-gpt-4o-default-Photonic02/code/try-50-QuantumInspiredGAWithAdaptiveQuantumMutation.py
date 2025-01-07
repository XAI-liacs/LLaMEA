import numpy as np

class QuantumInspiredGAWithAdaptiveQuantumMutation:
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
    
    def quantum_potential_well_mutation(self, individual):
        # Apply quantum potential well mutation to encourage exploration
        quantum_well = np.random.normal(0, self.mutation_factor, self.dim)
        mutated_individual = individual + quantum_well
        return mutated_individual
    
    def genetic_algorithm_step(self, func, target_idx, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        selected_indices = np.random.choice(self.population_size, 2, replace=False)
        parent1, parent2 = self.positions[selected_indices]
        
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(crossover_mask, parent1, parent2)
        
        mutated_offspring = self.quantum_potential_well_mutation(offspring)
        mutated_offspring = np.clip(mutated_offspring, lb, ub)
        
        return mutated_offspring, func(mutated_offspring)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.genetic_algorithm_step(func, i, evaluations)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                # Adaptive mutation factor based on convergence speed
                self.mutation_factor = self.initial_mutation_factor * (1 + (self.best_score - trial_score) / (self.best_score + 1e-9))
        
        # Final optimization with enhanced quantum mutation
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            enhanced_mutated_position = self.quantum_potential_well_mutation(self.positions[i])
            enhanced_mutated_position = np.clip(enhanced_mutated_position, func.bounds.lb, func.bounds.ub)
            enhanced_mutated_score = func(enhanced_mutated_position)
            evaluations += 1
            
            if enhanced_mutated_score < self.best_score:
                self.best_score = enhanced_mutated_score
                self.best_position = enhanced_mutated_position