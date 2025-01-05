import numpy as np

class Hybrid_DE_Adaptive_Quantum_Pso_Neighborhood_Learning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased for diverse sampling
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.6
        self.q_factor = 0.8
        self.gaussian_scale = 0.15
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptive_rate = 0.98
        self.temperature = 1.0
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            neighborhood_size = int(self.population_size / 3)  # Dynamic restructuring
            for i in range(self.population_size):
                # Differential Evolution mutation
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant_vector = position[indices[0]] + self.mutation_factor * (position[indices[1]] - position[indices[2]])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, position[i])

                # PSO update
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim) +
                                annealing_factor * np.random.uniform(-1, 1, self.dim))
                position[i] = np.clip(position[i], lb, ub)

                # Neighborhood learning
                neighborhood_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                for neighbor in neighborhood_indices:
                    local_best = min(personal_best_value[neighbor], personal_best_value[i])
                    if local_best < personal_best_value[i]:
                        position[i] = personal_best_position[neighbor]
                        break

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

        return global_best_position, global_best_value