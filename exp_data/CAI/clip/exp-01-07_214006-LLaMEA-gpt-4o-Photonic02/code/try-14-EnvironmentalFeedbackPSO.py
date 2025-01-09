import numpy as np

class EnvironmentalFeedbackPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        velocities = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) / 20
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        self.current_evaluations += self.population_size

        while self.current_evaluations < self.budget:
            # Calculate diversity as the standard deviation of the population
            diversity = np.mean(np.std(population, axis=0))

            for i in range(self.population_size):
                self.w = 0.5 + 0.5 * (1 - diversity)  # Adjust inertia based on diversity
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = self.w * velocities[i] + \
                                self.c1 * r1 * (personal_best_positions[i] - population[i]) + \
                                self.c2 * r2 * (global_best_position - population[i])
                population[i] += velocities[i]
                population[i] = np.clip(population[i], bounds[:,0], bounds[:,1])

                current_fitness = func(population[i])
                self.current_evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[global_best_index]:
                        global_best_position = population[i]
                        global_best_index = i

            # Dynamically adjust cognitive and social components based on progress
            progress = 1 - (self.current_evaluations / self.budget)
            self.c1 = 2.0 - progress * 1.0  # Increase cognitive component as progress decreases
            self.c2 = 1.0 + progress * 1.0  # Decrease social component as progress decreases

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]