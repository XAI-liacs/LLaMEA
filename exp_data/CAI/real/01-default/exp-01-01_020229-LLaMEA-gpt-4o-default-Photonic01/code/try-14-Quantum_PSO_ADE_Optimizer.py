import numpy as np

class Quantum_PSO_ADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.q_factor = 0.9
        self.crossover_rate = 0.85

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
                velocity[i] = (self.w * velocity[i] + 
                               self.c1 * r1 * (personal_best_position[i] - position[i]) + 
                               self.c2 * r2 * (global_best_position - position[i]))
                position[i] += velocity[i] + self.q_factor * np.random.normal(size=self.dim)
                position[i] = np.clip(position[i], lb, ub)

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

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutation_factor = 0.5 + 0.5 * (np.random.rand() - 0.5)
                mutant_vector = global_best_position + mutation_factor * (personal_best_position[b] - personal_best_position[c])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                trial_vector = np.copy(position[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < personal_best_value[i]:
                    personal_best_position[i] = trial_vector
                    personal_best_value[i] = trial_value
                
                if trial_value < global_best_value:
                    global_best_position = trial_vector
                    global_best_value = trial_value

        return global_best_position, global_best_value