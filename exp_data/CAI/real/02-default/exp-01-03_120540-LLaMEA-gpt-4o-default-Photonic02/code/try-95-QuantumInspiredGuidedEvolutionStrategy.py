import numpy as np

class QuantumInspiredGuidedEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.mutation_strength = 0.1
        self.crossover_rate = 0.8
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_guided_mutation(self, individual, global_best):
        # Introduce quantum-inspired randomization and guide towards the global best
        random_flip = np.random.uniform(-1, 1, self.dim) * self.mutation_strength
        guided_move = (global_best - individual) * np.random.rand()
        mutated_individual = individual + random_flip + guided_move
        return mutated_individual
    
    def guided_evolution(self, func, target_idx):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b = np.random.choice(indices, 2, replace=False)
        
        trial_vector = np.copy(self.positions[target_idx])
        if np.random.rand() < self.crossover_rate:
            trial_vector = self.positions[a] + np.random.rand(self.dim) * (self.positions[b] - trial_vector)
        
        trial_vector = np.clip(trial_vector, lb, ub)
        return trial_vector, func(trial_vector)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.guided_evolution(func, i)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
        
        # Final quantum-inspired global optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            mutated_position = self.quantum_guided_mutation(self.positions[i], self.best_position)
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position