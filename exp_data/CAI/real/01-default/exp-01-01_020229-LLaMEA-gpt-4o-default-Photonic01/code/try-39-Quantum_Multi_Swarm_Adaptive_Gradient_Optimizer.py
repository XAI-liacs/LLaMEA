import numpy as np

class Quantum_Multi_Swarm_Adaptive_Gradient_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.swarm_count = 5
        self.alpha = 0.9
        self.beta = 0.4
        self.gamma = 0.1
        self.delta = 0.1
        self.epsilon = 1e-8

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_values = [np.array([func(p) for p in personal_best]) for personal_best in personal_best_positions]
        global_best_position = min((min(p_best, key=lambda p: func(p)) for p_best in personal_best_positions), key=lambda p: func(p))
        global_best_value = func(global_best_position)
        
        evaluations = self.population_size * self.swarm_count

        while evaluations < self.budget:
            for swarm_idx in range(self.swarm_count):
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(2)
                    velocities[swarm_idx][i] = (self.alpha * velocities[swarm_idx][i] +
                                                self.beta * r1 * (personal_best_positions[swarm_idx][i] - swarms[swarm_idx][i]) +
                                                self.gamma * r2 * (global_best_position - swarms[swarm_idx][i]))
                    
                    gradient = (func(swarms[swarm_idx][i] + self.epsilon) - func(swarms[swarm_idx][i])) / self.epsilon
                    adaptive_gradient = self.delta * gradient
                    swarms[swarm_idx][i] += velocities[swarm_idx][i] + adaptive_gradient
                    
                    swarms[swarm_idx][i] = np.clip(swarms[swarm_idx][i], lb, ub)

                    current_value = func(swarms[swarm_idx][i])
                    evaluations += 1

                    if current_value < personal_best_values[swarm_idx][i]:
                        personal_best_positions[swarm_idx][i] = swarms[swarm_idx][i]
                        personal_best_values[swarm_idx][i] = current_value

                    if current_value < global_best_value:
                        global_best_position = swarms[swarm_idx][i]
                        global_best_value = current_value

                    if evaluations >= self.budget:
                        break

        return global_best_position, global_best_value