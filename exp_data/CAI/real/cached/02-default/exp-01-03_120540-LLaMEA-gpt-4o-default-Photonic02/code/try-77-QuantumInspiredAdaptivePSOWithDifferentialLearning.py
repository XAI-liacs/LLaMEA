import numpy as np

class QuantumInspiredAdaptivePSOWithDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.best_positions = np.copy(self.positions)
    
    def quantum_inspired_update(self, position, global_best):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_shift = np.where(quantum_bit, 1, -1)
        return position + self.w * quantum_shift * (global_best - position)
    
    def adaptive_velocity_update(self, particle_idx):
        r1, r2 = np.random.rand(), np.random.rand()
        personal_best_attraction = self.c1 * r1 * (self.best_positions[particle_idx] - self.positions[particle_idx])
        global_best_attraction = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        self.velocities[particle_idx] = self.w * self.velocities[particle_idx] + personal_best_attraction + global_best_attraction
    
    def apply_differential_learning(self, position, bounds):
        lb, ub = bounds.lb, bounds.ub
        diff_learning_factor = 0.5
        a, b, c = np.random.choice(self.swarm_size, 3, replace=False)
        differential_position = self.positions[a] + diff_learning_factor * (self.positions[b] - self.positions[c])
        new_position = position + differential_position
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                self.adaptive_velocity_update(i)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                current_score = func(self.positions[i])
                evaluations += 1

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]
                
                if current_score < func(self.best_positions[i]):
                    self.best_positions[i] = self.positions[i]
            
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                quantum_position = self.quantum_inspired_update(self.positions[i], self.global_best_position)
                quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
                quantum_score = func(quantum_position)
                evaluations += 1

                if quantum_score < self.global_best_score:
                    self.global_best_score = quantum_score
                    self.global_best_position = quantum_position

                diff_position = self.apply_differential_learning(self.positions[i], func.bounds)
                diff_score = func(diff_position)
                evaluations += 1

                if diff_score < self.global_best_score:
                    self.global_best_score = diff_score
                    self.global_best_position = diff_position