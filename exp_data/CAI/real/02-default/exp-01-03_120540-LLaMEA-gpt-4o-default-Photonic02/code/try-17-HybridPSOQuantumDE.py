import numpy as np

class HybridPSOQuantumDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def quantum_inspired_mutation(self, individual):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_individual = individual + self.mutation_factor * quantum_flip
        return mutated_individual
    
    def pso_update(self, func, idx):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (self.inertia_weight * self.velocities[idx] +
                                self.cognitive_weight * r1 * (self.personal_best_positions[idx] - self.positions[idx]) +
                                self.social_weight * r2 * (self.best_position - self.positions[idx]))
        
        self.positions[idx] = np.clip(self.positions[idx] + self.velocities[idx], func.bounds.lb, func.bounds.ub)
        current_score = func(self.positions[idx])
        
        if current_score < self.personal_best_scores[idx]:
            self.personal_best_scores[idx] = current_score
            self.personal_best_positions[idx] = self.positions[idx].copy()
    
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
                
                # PSO update
                self.pso_update(func, i)
                evaluations += 1
                
                if self.personal_best_scores[i] < self.best_score:
                    self.best_score = self.personal_best_scores[i]
                    self.best_position = self.personal_best_positions[i].copy()
                
                # Differential Evolution update
                trial_vector, trial_score = self.differential_evolution(func, i)
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                evaluations += 1

        # Final Quantum-inspired mutation
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