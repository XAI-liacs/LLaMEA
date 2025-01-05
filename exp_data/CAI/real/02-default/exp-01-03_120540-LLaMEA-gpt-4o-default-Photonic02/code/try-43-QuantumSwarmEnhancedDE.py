import numpy as np

class QuantumSwarmEnhancedDE:
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
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
    
    def quantum_swarm_update(self, individual, personal_best):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        velocity_update = (self.inertia_weight * self.velocities[individual] + 
                           self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best - self.positions[individual]) +
                           self.social_coefficient * np.random.rand(self.dim) * (self.global_best_position - self.positions[individual]))
        quantum_influence = quantum_flip * self.mutation_factor
        new_velocity = velocity_update + quantum_influence
        return new_velocity
    
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
        personal_bests = self.positions.copy()
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution step
                trial_vector, trial_score = self.differential_evolution(func, i)
                evaluations += 1

                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector

                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                    if trial_score < personal_best_scores[i]:
                        personal_bests[i] = trial_vector
                        personal_best_scores[i] = trial_score

                # Quantum-Swarm update
                self.velocities[i] = self.quantum_swarm_update(i, personal_bests[i])
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                # Update global best
                if personal_best_scores[i] < self.global_best_score:
                    self.global_best_score = personal_best_scores[i]
                    self.global_best_position = personal_bests[i]