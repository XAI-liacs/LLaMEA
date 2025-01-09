import numpy as np

class DynamicAdaptiveLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        def levy_flight(Lambda):
            sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                     (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
            u = np.random.normal(0, sigma, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / abs(v) ** (1 / Lambda)
            return step

        num_agents = 5 + int(0.1 * self.dim)
        inertia_weight = 0.9
        inertia_weight_min = 0.4
        cognitive_coefficient = 2.0
        social_coefficient = 2.0

        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (num_agents, self.dim))
        velocities = np.random.uniform(-1, 1, (num_agents, self.dim))

        fitness = np.apply_along_axis(func, 1, positions)
        self.eval_count += num_agents

        best_agent_idx = np.argmin(fitness)
        global_best_position = positions[best_agent_idx].copy()
        global_best_value = fitness[best_agent_idx]

        personal_best_positions = positions.copy()
        personal_best_values = fitness.copy()

        while self.eval_count < self.budget:
            for i in range(num_agents):
                # Adjust inertia weight dynamically
                inertia_weight = inertia_weight_min + (0.5 * (self.budget - self.eval_count) / self.budget)

                # Update velocity using PSO with Lévy Flight
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coefficient * r2 * (global_best_position - positions[i])
                levy_component = 0.01 * levy_flight(1.5)
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component + levy_component

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

                # Update global best
                if current_fitness < global_best_value:
                    global_best_position = positions[i]
                    global_best_value = current_fitness

                if self.eval_count >= self.budget:
                    break

        return global_best_position