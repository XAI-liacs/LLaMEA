import numpy as np

class AdaptiveSwarmIntelligence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 15 * dim
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.pareto_front_weight = 0.4
        self.velocities = np.random.rand(self.swarm_size, self.dim)
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = lb + np.random.rand(self.swarm_size, self.dim) * (ub - lb)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(ind) for ind in positions])
        evaluations = self.swarm_size
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2, r3 = np.random.rand(3)
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                    + self.social_constant * r2 * (global_best_position - positions[i])
                )
                
                pareto_influence = self.pareto_front_influence(positions, personal_best_fitness)
                self.velocities[i] += self.pareto_front_weight * r3 * pareto_influence

                # Update positions and clip within bounds
                positions[i] += self.velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Evaluate new positions
                new_fitness = func(positions[i])
                evaluations += 1

                # Update personal bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = positions[i]

                # Update global best
                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]

    def pareto_front_influence(self, positions, fitness):
        norms = np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)
        max_norm = np.max(norms)
        influence = norms / (max_norm + 1e-9)
        return influence