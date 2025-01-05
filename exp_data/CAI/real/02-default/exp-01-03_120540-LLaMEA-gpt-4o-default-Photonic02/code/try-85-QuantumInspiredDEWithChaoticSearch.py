import numpy as np

class QuantumInspiredDEWithChaoticSearch:
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
        self.chaos_sequence = self.generate_chaos_sequence(self.budget)
    
    def generate_chaos_sequence(self, length):
        # Generate a chaotic sequence using Logistic Map
        sequence = np.zeros(length)
        sequence[0] = 0.7  # initial value
        for i in range(1, length):
            sequence[i] = 4.0 * sequence[i - 1] * (1 - sequence[i - 1])
        return sequence

    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_inspired_mutation(self, individual):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_individual = individual + self.mutation_factor * quantum_flip
        return mutated_individual
    
    def chaotic_search(self, individual, iteration, bounds):
        # Perturb individual using chaotic sequence
        chaos_factor = self.chaos_sequence[iteration] * (bounds.ub - bounds.lb)
        return np.clip(individual + chaos_factor * (np.random.rand(self.dim) - 0.5), bounds.lb, bounds.ub)
    
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
        
        # Chaotic search for final optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            chaotic_position = self.chaotic_search(self.positions[i], evaluations, func.bounds)
            chaotic_score = func(chaotic_position)
            evaluations += 1
            
            if chaotic_score < self.best_score:
                self.best_score = chaotic_score
                self.best_position = chaotic_position