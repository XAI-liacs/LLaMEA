import numpy as np
import scipy.stats

class Adaptive_Quantum_PSO_Levy_LocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.729  # Inertia weight
        self.q_factor = 0.9
        self.alpha = 1.5  # Parameter for Lévy flight
        self.local_search_chance = 0.3
        self.ls_scale = 0.02  # Scale of local search

    def levy_flight(self):
        return scipy.stats.levy_stable.rvs(self.alpha, 0, size=self.dim)

    def local_search(self, position, lb, ub):
        perturbation = np.random.normal(0, self.ls_scale, size=position.shape)
        return np.clip(position + perturbation, lb, ub)

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

                # Apply Levy flight for exploration
                if np.random.rand() < 0.5:
                    position[i] += self.levy_flight() * self.q_factor

                position[i] += velocity[i]
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

            # Local Search Mechanism
            if np.random.rand() < self.local_search_chance:
                for i in range(self.population_size):
                    new_position = self.local_search(personal_best_position[i], lb, ub)
                    new_value = func(new_position)
                    evaluations += 1

                    if new_value < personal_best_value[i]:
                        personal_best_position[i] = new_position
                        personal_best_value[i] = new_value

                    if new_value < global_best_value:
                        global_best_position = new_position
                        global_best_value = new_value

                    if evaluations >= self.budget:
                        break

        return global_best_position, global_best_value