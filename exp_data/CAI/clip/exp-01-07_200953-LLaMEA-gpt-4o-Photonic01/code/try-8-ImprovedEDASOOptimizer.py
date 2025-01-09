import numpy as np

class ImprovedEDASOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 5 + int(0.1 * self.dim)
        exploration_factor = 0.9
        exploitation_factor = 0.1
        inertia_weight_min = 0.4
        inertia_weight_max = 0.9
        alpha = 0.5  # Coefficient for dynamic adaptation

        # Initialize position and velocity for each agent
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (num_agents, self.dim))
        velocities = np.random.uniform(-1, 1, (num_agents, self.dim))

        # Evaluate initial positions
        fitness = np.apply_along_axis(func, 1, positions)
        self.eval_count += num_agents

        best_agent_idx = np.argmin(fitness)
        global_best_position = positions[best_agent_idx].copy()
        global_best_value = fitness[best_agent_idx]

        personal_best_positions = positions.copy()
        personal_best_values = fitness.copy()

        while self.eval_count < self.budget:
            # Update inertia weight dynamically
            inertia_weight = inertia_weight_max - (inertia_weight_max - inertia_weight_min) * ((self.eval_count + 1) / self.budget)**0.95  # Changed line

            for i in range(num_agents):
                # Update velocity
                r1, r2 = np.random.rand(2)
                cognitive_component = exploration_factor * r1 * (personal_best_positions[i] - positions[i])
                social_component = exploitation_factor * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                # Introduce opposition-based learning
                opposite_position = lb + ub - positions[i]
                opposite_fitness = func(opposite_position)
                self.eval_count += 1

                if opposite_fitness < personal_best_values[i]:
                    positions[i] = opposite_position
                    personal_best_values[i] = opposite_fitness

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)  # ensure within bounds

                # Evaluate current position
                current_fitness = func(positions[i])
                self.eval_count += 1

                # Update personal bests
                if current_fitness < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_fitness

                # Update global best
                if current_fitness < global_best_value:
                    global_best_position = positions[i]
                    global_best_value = current_fitness

                if self.eval_count >= self.budget:
                    break

            # Cooperative learning using average of personal bests
            average_personal_best = np.mean(personal_best_positions, axis=0)
            for i in range(num_agents):
                if np.random.rand() < 0.1:  # 10% chance to use the average strategy
                    r3 = np.random.rand()
                    positions[i] = r3 * average_personal_best + (1 - r3) * positions[i]
                    positions[i] = np.clip(positions[i], lb, ub)
                    current_fitness = func(positions[i])
                    self.eval_count += 1
                    if current_fitness < personal_best_values[i]:
                        personal_best_positions[i] = positions[i]
                        personal_best_values[i] = current_fitness
                    if current_fitness < global_best_value:
                        global_best_position = positions[i]
                        global_best_value = current_fitness

            # Adaptive parameters
            exploration_factor = alpha * (1 - self.eval_count / self.budget)
            exploitation_factor = 1 - exploration_factor

        return global_best_position