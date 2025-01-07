import numpy as np

class MultiMemoryAdaptiveParticleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.memory_coeff = 0.5
        self.decay_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position_population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocity_population = np.random.rand(self.population_size, self.dim) * (ub - lb) * 0.1
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size

        personal_best_positions = np.copy(position_population)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = position_population[global_best_index]

        historical_best_positions = np.copy(personal_best_positions)
        historical_best_fitness = np.copy(personal_best_fitness)

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                inertia = self.inertia_weight * velocity_population[i]
                cognitive = self.cognitive_coeff * r1 * (personal_best_positions[i] - position_population[i])
                social = self.social_coeff * r2 * (global_best_position - position_population[i])

                memory_r = np.random.rand()
                memory_effect = self.memory_coeff * memory_r * (historical_best_positions[i] - position_population[i])
                velocity_population[i] = inertia + cognitive + social + memory_effect

                position_population[i] += velocity_population[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                if new_fitness < historical_best_fitness[i]:
                    historical_best_fitness[i] = new_fitness
                    historical_best_positions[i] = position_population[i]

                if new_fitness < fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position_population[i]

                if evaluations >= self.budget:
                    break

            self.inertia_weight *= self.decay_rate

        return global_best_position, fitness[global_best_index]