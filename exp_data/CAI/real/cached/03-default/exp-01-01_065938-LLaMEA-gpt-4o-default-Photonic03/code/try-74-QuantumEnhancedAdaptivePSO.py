import numpy as np

class QuantumEnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.c1, self.c2 = 2.0, 2.0
        self.w_min, self.w_max = 0.4, 0.9
        self.success_rates = [0.5, 0.5]
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        positions = lb + (ub - lb) * np.cos(np.pi * population_quantum)
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(x) for x in positions])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) + 
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                fitness = func(positions[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i]
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = positions[i]

        return global_best_position