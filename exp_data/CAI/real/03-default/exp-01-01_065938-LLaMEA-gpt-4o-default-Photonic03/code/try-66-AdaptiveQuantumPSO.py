import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.w_min, self.w_max = 0.4, 0.9
        self.c1, self.c2 = 2.0, 2.0
        self.quantum_prob = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        velocities = np.zeros_like(positions)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(x) for x in positions])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        evaluations = self.num_particles

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
           
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] 
                                + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                                + self.c2 * r2 * (global_best_position - positions[i]))
                
                if np.random.rand() < self.quantum_prob:
                    quantum_delta = np.random.uniform(-0.5, 0.5, self.dim)
                    positions[i] = global_best_position + quantum_delta * (ub - lb)
                else:
                    positions[i] += velocities[i]
                
                positions[i] = np.clip(positions[i], lb, ub)
                
                fitness = func(positions[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i]

                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = positions[i]

        return global_best_position