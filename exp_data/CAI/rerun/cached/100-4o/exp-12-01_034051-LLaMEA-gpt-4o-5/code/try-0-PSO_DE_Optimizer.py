import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.differential_weight = 0.8
        self.crossover_prob = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(0)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_value = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i]

                if evaluations >= self.budget:
                    break

            # Update velocity and position for PSO
            inertia_component = self.inertia_weight * velocity
            cognitive_component = self.cognitive_coef * np.random.rand(self.dim) * (personal_best_position - position)
            social_component = self.social_coef * np.random.rand(self.dim) * (global_best_position - position)
            velocity = inertia_component + cognitive_component + social_component
            position = np.clip(position + velocity, self.lower_bound, self.upper_bound)

            # Differential Evolution step
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = np.clip(personal_best_position[a] + self.differential_weight * (personal_best_position[b] - personal_best_position[c]), self.lower_bound, self.upper_bound)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_prob else position[i][j] for j in range(self.dim)])
                
                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < personal_best_value[i]:
                    personal_best_value[i] = trial_value
                    personal_best_position[i] = trial_vector

                if trial_value < global_best_value:
                    global_best_value = trial_value
                    global_best_position = trial_vector

        return global_best_position, global_best_value