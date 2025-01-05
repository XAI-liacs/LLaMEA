import numpy as np

class Adaptive_Evolutionary_Quantum_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.mutation_rate = 0.05
        self.q_factor = 1.0
        self.evolutionary_pressure = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in position])
        personal_best_position = np.copy(position)
        personal_best_value = np.copy(fitness)
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive Inertia Weight
            self.w = 0.9 - 0.7 * (evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                # Apply quantum tunneling effect
                tunneling_effect = self.q_factor * np.random.normal(size=self.dim)
                position[i] += velocity[i] + tunneling_effect
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

            # Evolutionary Selection and Mutation
            sorted_indices = np.argsort(personal_best_value)
            survivors = sorted_indices[:int(self.population_size * (1 - self.evolutionary_pressure))]
            offspring_size = self.population_size - len(survivors)
            offspring = personal_best_position[np.random.choice(survivors, offspring_size)]
            mutation_mask = np.random.rand(offspring_size, self.dim) < self.mutation_rate
            offspring += mutation_mask * np.random.laplace(size=(offspring_size, self.dim))
            offspring = np.clip(offspring, lb, ub)

            # Replace least fit individuals with offspring
            position[sorted_indices[-offspring_size:]] = offspring
            fitness[sorted_indices[-offspring_size:]] = [func(p) for p in offspring]
            evaluations += offspring_size

            for i in range(offspring_size):
                if fitness[sorted_indices[-i-1]] < personal_best_value[sorted_indices[-i-1]]:
                    personal_best_position[sorted_indices[-i-1]] = position[sorted_indices[-i-1]]
                    personal_best_value[sorted_indices[-i-1]] = fitness[sorted_indices[-i-1]]

                if fitness[sorted_indices[-i-1]] < global_best_value:
                    global_best_position = position[sorted_indices[-i-1]]
                    global_best_value = fitness[sorted_indices[-i-1]]

        return global_best_position, global_best_value