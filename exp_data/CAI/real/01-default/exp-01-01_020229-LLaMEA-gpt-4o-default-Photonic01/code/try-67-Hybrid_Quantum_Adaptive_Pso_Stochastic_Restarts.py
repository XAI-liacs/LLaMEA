import numpy as np

class Hybrid_Quantum_Adaptive_Pso_Stochastic_Restarts:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Further increased for diverse exploration
        self.c1 = 1.5  # Adjusted cognitive and social factors
        self.c2 = 2.5
        self.w = 0.9  # Increased initial inertia for better exploration
        self.w_min = 0.4  # Minimum inertia weight for better convergence
        self.alpha = 0.1  # Coefficient for stochastic restarts
        self.q_factor = 0.8
        self.temperature = 1.0  # Initial temperature for simulated annealing
        self.decrement_factor = 0.95  # Temperature decrement factor
        self.random_seed = 42

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(self.random_seed)

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
                self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
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

            self.temperature *= self.decrement_factor  # Simulated annealing effect

            # Stochastic Restart Mechanism
            if np.random.rand() < self.alpha:
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