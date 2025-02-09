import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.alpha = 0.75  # Quantum parameter
        self.beta = 0.75   # Adaptive parameter

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_fitness = np.array([func(ind) for ind in swarm])
        global_best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]
        evaluations = self.swarm_size

        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Quantum-inspired update
                local_attractor = personal_best[i] + self.alpha * (global_best - personal_best[i])
                velocities[i] = self.beta * velocities[i] + np.random.uniform(low=lb, high=ub) * (local_attractor - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)
                
                # Evaluate new positions
                fitness = func(swarm[i])
                evaluations += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = swarm[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = swarm[i]
                        global_best_fitness = fitness

                if evaluations >= self.budget:
                    break

            # Adaptive parameter update
            self.beta = 0.5 + (np.var(personal_best_fitness) / np.mean(personal_best_fitness)) * 0.5
            self.beta = np.clip(self.beta, 0.1, 1.0)

        return global_best