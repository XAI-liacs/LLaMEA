import numpy as np

class Quantum_Inspired_Hybrid_Evolutionary_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.1  # Quantum potential well depth
        self.beta = 0.9  # Differential evolution factor
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9
        self.gamma = 0.5  # Chaos influence factor
        self.delta = 0.2  # Chaos scale factor

    def chaotic_sequence(self, x):
        return self.gamma * x * (1 - x)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize population
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size
        chaotic_var = np.random.rand()

        while evaluations < self.budget:
            # Quantum potential well influence
            q_potential = self.alpha * np.random.uniform(-1, 1, (self.population_size, self.dim))
            q_positions = np.clip(position + q_potential, lb, ub)

            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = position[indices]
                mutant_vector = x1 + self.beta * (x2 - x3)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, position[i])
                trial_vector = np.clip(trial_vector, lb, ub)
                
                # Evaluate candidate solutions
                q_value = func(q_positions[i])
                trial_value = func(trial_vector)
                evaluations += 2
                
                # Update personal best
                if q_value < personal_best_value[i]:
                    personal_best_position[i] = q_positions[i]
                    personal_best_value[i] = q_value
                
                if trial_value < personal_best_value[i]:
                    personal_best_position[i] = trial_vector
                    personal_best_value[i] = trial_value

                # Update global best
                if personal_best_value[i] < global_best_value:
                    global_best_position = personal_best_position[i]
                    global_best_value = personal_best_value[i]

                if evaluations >= self.budget:
                    break

            # Apply chaotic influence
            chaotic_var = self.chaotic_sequence(chaotic_var)
            chaos_impact = self.delta * (chaotic_var - 0.5) * (ub - lb)
            position += chaos_impact
            position = np.clip(position, lb, ub)

        return global_best_position, global_best_value