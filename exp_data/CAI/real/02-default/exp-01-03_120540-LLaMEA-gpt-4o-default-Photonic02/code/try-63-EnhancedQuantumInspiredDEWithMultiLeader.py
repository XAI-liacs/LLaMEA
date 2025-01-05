import numpy as np

class EnhancedQuantumInspiredDEWithMultiLeader:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_positions = []
        self.best_scores = []
        self.initial_mutation_factor = 0.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = [float('inf')] * self.population_size
    
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
        
        # Select multiple leaders for diverse guidance
        leader_idx = np.random.choice(range(self.population_size), 2, replace=False)
        leader_1, leader_2 = self.best_positions[leader_idx[0]], self.best_positions[leader_idx[1]]
        
        mutant_vector = leader_1 + self.mutation_factor * (self.positions[b] - self.positions[c]) + self.mutation_factor * (leader_2 - self.positions[a])
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
                
                if trial_score < self.best_scores[i]:
                    self.best_scores[i] = trial_score
                    self.best_positions[i] = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                self.mutation_factor = self.initial_mutation_factor * (1 - evaluations / self.budget)
        
        # Quantum-inspired global search for final optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            mutated_position = self.quantum_inspired_mutation(self.positions[i])
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < min(self.best_scores):
                best_idx = np.argmin(self.best_scores)
                self.best_scores[best_idx] = mutated_score
                self.best_positions[best_idx] = mutated_position