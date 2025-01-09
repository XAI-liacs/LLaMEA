import numpy as np

class HybridPSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 5 + int(0.1 * self.dim)
        inertia_weight = 0.9
        cognitive_coefficient = 1.75
        social_coefficient = 1.35
        F = 0.6
        CR = 0.85

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
            for i in range(num_agents):
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coefficient * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                inertia_weight = max(0.4, inertia_weight * 0.98)

                # Update position and apply Lévy flight
                positions[i] += velocities[i]
                if np.random.rand() < 0.5:  # Apply Lévy flight with a probability
                    levy_step = np.random.standard_cauchy(self.dim) * 0.01
                    positions[i] += levy_step
                positions[i] = np.clip(positions[i], lb, ub)
                
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

            for i in range(num_agents):
                if np.random.rand() < CR:
                    indices = np.random.choice(num_agents, 3, replace=False)
                    x1, x2, x3 = positions[indices]
                    mutant_vector = x1 + F * (x2 - x3)
                    trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, positions[i])
                    trial_vector = np.clip(trial_vector, lb, ub)

                    trial_fitness = func(trial_vector)
                    self.eval_count += 1

                    if trial_fitness < personal_best_values[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_values[i] = trial_fitness

                    if trial_fitness < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_fitness

                    if self.eval_count >= self.budget:
                        break

            for i in range(num_agents):
                perturbation = np.random.normal(0, np.random.uniform(0.05, 0.15), self.dim)
                candidate_position = positions[i] + perturbation
                candidate_position = np.clip(candidate_position, lb, ub)
                candidate_fitness = func(candidate_position)
                self.eval_count += 1
                if candidate_fitness < personal_best_values[i]:
                    personal_best_positions[i] = candidate_position
                    personal_best_values[i] = candidate_fitness
                if candidate_fitness < global_best_value:
                    global_best_position = candidate_position
                    global_best_value = candidate_fitness

        return global_best_position