import numpy as np

class Enhanced_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.initial_w = 0.9
        self.final_w = 0.4
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.success_threshold = 0.2

    def adaptive_inertia_weight(self, evaluations):
        return (self.initial_w - self.final_w) * (1 - evaluations / self.budget) + self.final_w

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
            success_count = 0
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Adaptive Inertia Weight
                w = self.adaptive_inertia_weight(evaluations)
                
                # PSO Update
                r1, r2 = np.random.rand(2)
                velocity[i] = (w * velocity[i] + 
                               self.c1 * r1 * (personal_best_position[i] - position[i]) + 
                               self.c2 * r2 * (global_best_position - position[i]))
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                    success_count += 1
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

            # Success-based parameter adaptation
            if success_count / self.population_size > self.success_threshold:
                self.crossover_rate *= 1.05
                self.mutation_factor *= 1.05
            else:
                self.crossover_rate *= 0.95
                self.mutation_factor *= 0.95

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = personal_best_position[a] + self.mutation_factor * (personal_best_position[b] - personal_best_position[c])
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