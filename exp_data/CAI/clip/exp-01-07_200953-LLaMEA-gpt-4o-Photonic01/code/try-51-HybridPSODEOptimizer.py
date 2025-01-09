import numpy as np

class HybridPSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 5 + int(0.1 * self.dim)
        inertia_weight = 0.9  # Adjusted initial inertia weight
        cognitive_coefficient = 1.5
        social_coefficient = 1.5
        F = 0.5  # Differential evolution scaling factor
        CR = 0.9  # Crossover probability

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
            # Dynamically adjust the number of agents
            num_agents = max(5, num_agents - (self.eval_count // (self.budget / num_agents)))

            for i in range(num_agents):
                # Update velocity using PSO
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coefficient * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

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

            # Differential Evolution-inspired mutation
            for i in range(num_agents):
                if np.random.rand() < CR:
                    indices = np.random.choice(num_agents, 3, replace=False)
                    x1, x2, x3 = positions[indices]
                    mutant_vector = x1 + F * (x2 - x3)
                    trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, positions[i])
                    trial_vector = np.clip(trial_vector, lb, ub)

                    trial_fitness = func(trial_vector)
                    self.eval_count += 1

                    # Update personal bests and global best for trial vector
                    if trial_fitness < personal_best_values[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_values[i] = trial_fitness

                    if trial_fitness < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_fitness

                    if self.eval_count >= self.budget:
                        break

            inertia_weight = 0.4 + (0.5 * (self.budget - self.eval_count) / self.budget)  # Adapt inertia weight

        return global_best_position