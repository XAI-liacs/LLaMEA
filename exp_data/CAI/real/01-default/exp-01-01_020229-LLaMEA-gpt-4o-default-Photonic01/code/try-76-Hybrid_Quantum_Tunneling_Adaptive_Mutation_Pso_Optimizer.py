import numpy as np

class Hybrid_Quantum_Tunneling_Adaptive_Mutation_Pso_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased to enhance exploration
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.quantum_factor = 0.8
        self.gaussian_scale = 0.1
        self.initial_reset_chance = 0.05
        self.temperature = 1.0
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= 0.99  # Slight adaptive decrease in inertia
                
                # Velocity update with quantum tunneling effect
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]) +
                               self.quantum_factor * np.random.normal(scale=self.gaussian_scale, size=self.dim))
                
                # Position update with adaptive differential mutation
                mutant_vector = position[i] + self.mutation_factor * (personal_best_position[i] - position[i])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, position[i])
                position[i] = np.clip(trial_vector + velocity[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1
                
                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            # Random search enhancement based on reset chance
            if np.random.rand() < self.initial_reset_chance * (1 - (evaluations / self.budget)):
                random_index = np.random.randint(self.population_size)
                position[random_index] = np.random.uniform(lb, ub, self.dim)
                current_value = func(position[random_index])
                evaluations += 1
                
                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = position[random_index]
                    personal_best_value[random_index] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value
        
        return global_best_position, global_best_value