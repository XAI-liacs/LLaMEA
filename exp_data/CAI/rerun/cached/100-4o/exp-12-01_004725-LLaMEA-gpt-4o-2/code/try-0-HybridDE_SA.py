import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.temperature = 100.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        eval_count = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        eval_count += self.population_size

        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Differential Evolution Mutation
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]
                donor_vector = x_r1 + self.F * (x_r2 - x_r3)
                donor_vector = np.clip(donor_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_points] = donor_vector[crossover_points]

                # Simulated Annealing Acceptance
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    acceptance_probability = np.exp((fitness[i] - trial_fitness) / self.temperature)
                    if np.random.rand() < acceptance_probability:
                        new_population[i] = trial_vector
                        fitness[i] = trial_fitness

            population = new_population
            self.temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_solution, best_fitness