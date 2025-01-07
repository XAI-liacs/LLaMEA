import numpy as np

class Integrated_Quantum_Pso_Adaptive_Inertia_Elite_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.q_factor = 0.9
        self.gaussian_scale = 0.1
        self.reset_chance = 0.05
        self.adaptive_rate = 0.99
        self.elite_fraction = 0.1
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize positions and velocities
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate  # Gradually decrease inertia weight
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                adaptive_gaussian_scale = self.gaussian_scale * (1 - self.evaluations / self.budget)
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim))
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                self.evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value

                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if self.evaluations >= self.budget:
                    break

            # Elite Opposition-Based Learning
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(personal_best_value)[:elite_count]
            for idx in elite_indices:
                opposite_position = lb + ub - personal_best_position[idx]
                opposite_position = np.clip(opposite_position, lb, ub)
                opposite_value = func(opposite_position)
                self.evaluations += 1

                if opposite_value < personal_best_value[idx]:
                    personal_best_position[idx] = opposite_position
                    personal_best_value[idx] = opposite_value

                if opposite_value < global_best_value:
                    global_best_position = opposite_position
                    global_best_value = opposite_value

                if self.evaluations >= self.budget:
                    break

            # Global Position Reset Mechanism
            if np.random.rand() < self.reset_chance:
                random_index = np.random.randint(self.population_size)
                position[random_index] = np.random.uniform(lb, ub, self.dim)
                current_value = func(position[random_index])
                self.evaluations += 1

                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = position[random_index]
                    personal_best_value[random_index] = current_value

                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value

        return global_best_position, global_best_value