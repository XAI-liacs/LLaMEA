import numpy as np

class EnhancedEDASOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        num_agents = 5 + int(0.1 * self.dim)
        exploration_factor = 0.9
        exploitation_factor = 0.1
        inertia_weight_min = 0.4
        inertia_weight_max = 0.9
        alpha = 0.5

        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (num_agents, self.dim))
        velocities = np.random.uniform(-1, 1, (num_agents, self.dim))
        velocity_scaling_factor = 0.5

        fitness = np.apply_along_axis(func, 1, positions)
        self.eval_count += num_agents

        best_agent_idx = np.argmin(fitness)
        global_best_position = positions[best_agent_idx].copy()
        global_best_value = fitness[best_agent_idx]

        personal_best_positions = positions.copy()
        personal_best_values = fitness.copy()

        while self.eval_count < self.budget:
            inertia_weight = inertia_weight_max - (inertia_weight_max - inertia_weight_min) * ((self.eval_count + 1) / self.budget)

            for i in range(num_agents):
                r1, r2 = np.random.rand(2)
                cognitive_component = exploration_factor * r1 * (personal_best_positions[i] - positions[i])
                social_component = exploitation_factor * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                velocities[i] *= velocity_scaling_factor

                opposite_position = lb + ub - positions[i]
                opposite_fitness = func(opposite_position)
                self.eval_count += 1

                if opposite_fitness < personal_best_values[i]:
                    positions[i] = opposite_position
                    personal_best_values[i] = opposite_fitness

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                current_fitness = func(positions[i])
                self.eval_count += 1

                if current_fitness < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_fitness

                if current_fitness < global_best_value:
                    global_best_position = positions[i]
                    global_best_value = current_fitness

                if self.eval_count >= self.budget:
                    break

            tournament_size = max(2, num_agents // 3)
            selected_indices = np.random.choice(num_agents, tournament_size, replace=False)
            tournament_best_idx = min(selected_indices, key=lambda idx: personal_best_values[idx])
            global_best_position = personal_best_positions[tournament_best_idx]
            global_best_value = personal_best_values[tournament_best_idx]

            exploration_factor = alpha * (1 - self.eval_count / self.budget)
            exploitation_factor = 1 - exploration_factor

        return global_best_position