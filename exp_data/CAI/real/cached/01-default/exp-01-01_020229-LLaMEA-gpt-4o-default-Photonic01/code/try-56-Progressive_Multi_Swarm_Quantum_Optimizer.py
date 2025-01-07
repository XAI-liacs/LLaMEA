import numpy as np

class Progressive_Multi_Swarm_Quantum_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.8
        self.alpha = 0.9
        self.beta = 0.1
        self.q_sigma = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        swarms = [np.random.uniform(lb, ub, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarms[i]) for i in range(self.num_swarms)]
        personal_best_values = [np.array([func(p) for p in personal_best_positions[i]]) for i in range(self.num_swarms)]
        global_best_positions = [personal_best_positions[i][np.argmin(personal_best_values[i])] for i in range(self.num_swarms)]
        global_best_values = [np.min(personal_best_values[i]) for i in range(self.num_swarms)]

        evaluations = self.swarm_size * self.num_swarms

        while evaluations < self.budget:
            for swarm_index in range(self.num_swarms):
                for i in range(self.swarm_size):
                    r1, r2, r3 = np.random.rand(3)
                    velocities[swarm_index][i] = (
                        self.w * velocities[swarm_index][i] +
                        self.c1 * r1 * (personal_best_positions[swarm_index][i] - swarms[swarm_index][i]) +
                        self.c2 * r2 * (global_best_positions[swarm_index] - swarms[swarm_index][i])
                    )
                    
                    quantum_factor = self.q_sigma * np.random.standard_normal(size=self.dim)
                    swarms[swarm_index][i] += velocities[swarm_index][i] + quantum_factor
                    swarms[swarm_index][i] = np.clip(swarms[swarm_index][i], lb, ub)

                    current_value = func(swarms[swarm_index][i])
                    evaluations += 1

                    if current_value < personal_best_values[swarm_index][i]:
                        personal_best_positions[swarm_index][i] = swarms[swarm_index][i]
                        personal_best_values[swarm_index][i] = current_value

                    if current_value < global_best_values[swarm_index]:
                        global_best_positions[swarm_index] = swarms[swarm_index][i]
                        global_best_values[swarm_index] = current_value

                    if evaluations >= self.budget:
                        break

            # Inter-swarm communication and hierarchical update
            if swarm_index < self.num_swarms - 1:
                next_swarm_index = swarm_index + 1
                if global_best_values[swarm_index] < global_best_values[next_swarm_index]:
                    global_best_positions[next_swarm_index] = global_best_positions[swarm_index]
                    global_best_values[next_swarm_index] = global_best_values[swarm_index]

            # Update velocities and positions adaptively
            self.w *= self.alpha
            self.q_sigma *= self.beta

        best_swarm_index = np.argmin(global_best_values)
        return global_best_positions[best_swarm_index], global_best_values[best_swarm_index]