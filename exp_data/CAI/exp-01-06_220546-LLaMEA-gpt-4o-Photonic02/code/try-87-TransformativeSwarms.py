import numpy as np

class TransformativeSwarms:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.local_search_prob = 0.3

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        velocity = np.zeros((self.population_size, self.dim))
        position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(ind) for ind in position])
        global_best_index = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_index]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            # Update velocity and position
            r1, r2 = np.random.rand(2)
            self.inertia = 0.9 - 0.4 * (eval_count / self.budget)  # Change: refined inertia variation
            self.social_coef = 1.5 * (1 + np.sin(eval_count / self.budget * np.pi))
            velocity = (self.inertia * velocity +
                        self.cognitive_coef * r1 * (personal_best_position - position) +
                        self.social_coef * r2 * (global_best_position - position))
            position += velocity
            position = np.clip(position, lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in position])
            eval_count += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if new_values[i] < personal_best_value[i]:
                    personal_best_value[i] = new_values[i]
                    personal_best_position[i] = position[i]
            global_best_index = np.argmin(personal_best_value)
            if personal_best_value[global_best_index] < func(global_best_position):
                global_best_position = personal_best_position[global_best_index]

            # Adaptive local search
            self.local_search_prob = 0.3 * (1 - (eval_count / self.budget))
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.normal(0, 0.1, self.dim)
                for i in range(self.population_size):
                    candidate = position[i] + perturbation
                    candidate = np.clip(candidate, lb, ub)
                    candidate_value = func(candidate)
                    eval_count += 1
                    if candidate_value < personal_best_value[i]:
                        personal_best_value[i] = candidate_value
                        personal_best_position[i] = candidate

        return global_best_position, func(global_best_position)