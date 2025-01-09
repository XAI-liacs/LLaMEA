import numpy as np

class AQPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 10 + int(0.15 * self.dim)
        inertia_weight = 0.7
        cognitive_coeff = 2.0
        social_coeff = 2.0
        quantum_factor = 0.5

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
                # Update velocity using quantum-inspired approach
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Implement quantum-inspired probability adaptation
                if np.random.rand() < quantum_factor:
                    quantum_position = lb[i] + np.random.rand() * (ub[i] - lb[i])
                    quantum_fitness = func(quantum_position)
                    self.eval_count += 1
                    if quantum_fitness < personal_best_values[i]:
                        positions[i] = quantum_position
                        personal_best_values[i] = quantum_fitness

                # Update position with hypercube strategy
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

            # Adaptive quantum factor adjustment
            quantum_factor = 0.5 * (1 - self.eval_count / self.budget)

        return global_best_position