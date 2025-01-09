import numpy as np

class QuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.inertia_weight = 0.9
        self.cognitive_param = 2.0
        self.social_param = 2.0
        self.quantum_param = 0.5
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        eval_count = self.swarm_size

        while eval_count < self.budget:
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i] +
                    self.cognitive_param * r1 * (personal_best[i] - swarm[i]) +
                    self.social_param * r2 * (global_best - swarm[i])
                )
                quantum_walk = self.quantum_param * np.random.uniform(lb, ub, self.dim)
                swarm[i] = np.where(np.random.rand(self.dim) < 0.5, swarm[i] + velocities[i], quantum_walk)
                swarm[i] = np.clip(swarm[i], lb, ub)

                current_fitness = func(swarm[i])
                eval_count += 1
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = swarm[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < global_best_fitness:
                        global_best = swarm[i]
                        global_best_fitness = current_fitness

            # Dynamically adjust inertia weight
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (eval_count / self.budget))

        return global_best