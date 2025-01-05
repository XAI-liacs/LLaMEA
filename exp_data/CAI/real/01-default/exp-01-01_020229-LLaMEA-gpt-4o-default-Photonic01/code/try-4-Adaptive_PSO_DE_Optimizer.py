import numpy as np

class Adaptive_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Number of particles
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w_min = 0.4
        self.w_max = 0.9
        self.mutation_factor_base = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize the swarm
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size
        iteration = 0

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            adaptive_mutation_factor = self.mutation_factor_base + (0.5 * np.sin(2 * np.pi * iteration / 20))

            for i in range(self.population_size):
                # PSO Update
                r1, r2 = np.random.rand(2)
                velocity[i] = (w * velocity[i] + 
                               self.c1 * r1 * (personal_best_position[i] - position[i]) + 
                               self.c2 * r2 * (global_best_position - position[i]))
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                # Evaluate
                current_value = func(position[i])
                evaluations += 1
                
                # Update personal best
                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                
                # Update global best
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            # Adaptive DE Variant incorporated into PSO
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = personal_best_position[a] + adaptive_mutation_factor * (personal_best_position[b] - personal_best_position[c])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                trial_vector = np.copy(position[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_value = func(trial_vector)
                evaluations += 1

                # If trial vector is better, replace the target
                if trial_value < personal_best_value[i]:
                    personal_best_position[i] = trial_vector
                    personal_best_value[i] = trial_value
                
                # Update global best after DE step
                if trial_value < global_best_value:
                    global_best_position = trial_vector
                    global_best_value = trial_value

            iteration += 1

        return global_best_position, global_best_value