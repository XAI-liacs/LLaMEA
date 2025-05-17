import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, swarm_size=30, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize the swarm and velocities
        swarm = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        # Initialize the best-known positions
        personal_best_positions = np.copy(swarm)
        personal_best_values = np.array([func(x) for x in swarm])
        
        # Global best position
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        evaluations = self.swarm_size
        
        while evaluations < self.budget:
            # Update velocities and positions
            for i in range(self.swarm_size):
                # PSO velocity update
                r1, r2 = np.random.rand(2)
                cognitive_velocity = r1 * (personal_best_positions[i] - swarm[i])
                social_velocity = r2 * (global_best_position - swarm[i])
                velocities[i] = 0.7 * velocities[i] + cognitive_velocity + social_velocity
                
                # Update position
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], -100, 100)
                
                # Evaluate the new position
                fitness = func(swarm[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_values[i] = fitness

                    # Update global best
                    if fitness < global_best_value:
                        global_best_position = swarm[i]
                        global_best_value = fitness

            # Differential Evolution Mutation and Crossover
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = [idx for idx in range(self.swarm_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = personal_best_positions[a] + self.F * (personal_best_positions[b] - personal_best_positions[c])
                mutant_vector = np.clip(mutant_vector, -100, 100)

                # Crossover
                trial_vector = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial_vector[j] = mutant_vector[j]

                # Selection
                fitness = func(trial_vector)
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_values[i] = fitness

                    # Update global best
                    if fitness < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = fitness

        return global_best_value, global_best_position