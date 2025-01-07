import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w_min, self.w_max = 0.4, 0.9
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.alpha_min, self.alpha_max = 0.5, 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        positions = lb + (ub - lb) * np.cos(np.pi * position_quantum)
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb) * 0.1
        fitness = np.array([func(x) for x in positions])
        
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                w = np.random.uniform(self.w_min, self.w_max)
                c1 = np.random.uniform(self.c1_min, self.c1_max)
                c2 = np.random.uniform(self.c2_min, self.c2_max)
                alpha = np.random.uniform(self.alpha_min, self.alpha_max)

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2 * (global_best_position - positions[i]))

                quantum_shift = np.random.uniform(-alpha, alpha, self.dim)
                new_position = positions[i] + velocities[i] + quantum_shift

                new_position = np.clip(new_position, lb, ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = new_position
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = new_position

            positions += velocities

        return global_best_position