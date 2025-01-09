import numpy as np

class HybridDynamicVelocityOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 5 + int(0.1 * self.dim)
        inertia_weight_min = 0.3
        inertia_weight_max = 0.9
        personal_coef = 2.0
        social_coef = 2.0
        adaptation_rate = 0.01

        # Initialize positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (num_agents, self.dim))
        velocities = np.random.uniform(-1, 1, (num_agents, self.dim))

        # Evaluate initial positions
        fitness = np.apply_along_axis(func, 1, positions)
        self.eval_count += num_agents

        best_global_idx = np.argmin(fitness)
        global_best_position = positions[best_global_idx].copy()
        global_best_value = fitness[best_global_idx]

        personal_best_positions = positions.copy()
        personal_best_values = fitness.copy()

        while self.eval_count < self.budget:
            # Dynamically adjust inertia weight
            inertia_weight = inertia_weight_max - (inertia_weight_max - inertia_weight_min) * (self.eval_count / self.budget)

            for i in range(num_agents):
                # Update velocity
                r1, r2 = np.random.rand(2)
                cognitive_component = personal_coef * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coef * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                # Dynamic velocity adjustment
                velocities[i] *= (1 - adaptation_rate) + 2 * adaptation_rate * np.random.rand()

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate current position
                current_fitness = func(positions[i])
                self.eval_count += 1

                # Update personal bests
                if current_fitness < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_fitness

                # Update global best if necessary
                if current_fitness < global_best_value:
                    global_best_position = positions[i]
                    global_best_value = current_fitness

                # Adaptive opposition-based learning
                if np.random.rand() < 0.2:  # 20% probability to use opposition-based strategy
                    opposite_position = lb + ub - positions[i]
                    opposite_fitness = func(opposite_position)
                    self.eval_count += 1

                    if opposite_fitness < personal_best_values[i]:
                        personal_best_positions[i] = opposite_position
                        personal_best_values[i] = opposite_fitness
                        if opposite_fitness < global_best_value:
                            global_best_position = opposite_position
                            global_best_value = opposite_fitness

                if self.eval_count >= self.budget:
                    break

        return global_best_position