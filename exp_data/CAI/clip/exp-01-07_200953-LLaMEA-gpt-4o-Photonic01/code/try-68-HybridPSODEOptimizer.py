import numpy as np

class HybridPSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        num_agents = 5 + int(0.1 * self.dim)
        initial_inertia_weight = 0.9
        final_inertia_weight = 0.4
        cognitive_coefficient = 1.5
        social_coefficient = 1.5
        F_init, F_final = 0.4, 0.9
        CR_init, CR_final = 0.6, 0.95
        
        # Initializing positions and velocities
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
            inertia_weight = initial_inertia_weight - (initial_inertia_weight - final_inertia_weight) * (self.eval_count / self.budget)

            for i in range(num_agents):
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coefficient * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

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

            F = F_init + (F_final - F_init) * (self.eval_count / self.budget)
            CR = CR_init + (CR_final - CR_init) * (self.eval_count / self.budget)

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
                        
        return global_best_position