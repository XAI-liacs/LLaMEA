import numpy as np

class QuantumInspiredDEWithAdaptiveQuantumTunneling:
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
        self.tunneling_probability = 0.1

    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adaptive_quantum_tunneling(self, individual, evaluations):
        probability = self.tunneling_probability * (1 - evaluations / self.budget)
        if np.random.rand() < probability:
            tunnel_shift = np.random.normal(0, 1, self.dim)
            return individual + tunnel_shift
        return individual

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
        
        # Adaptive quantum tunneling for enhanced exploration
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            tunneled_position = self.adaptive_quantum_tunneling(self.positions[i], evaluations)
            tunneled_position = np.clip(tunneled_position, func.bounds.lb, func.bounds.ub)
            tunneled_score = func(tunneled_position)
            evaluations += 1
            
            if tunneled_score < self.best_score:
                self.best_score = tunneled_score
                self.best_position = tunneled_position