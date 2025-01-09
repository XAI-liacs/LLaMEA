import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temp_initial = 100.0
        self.temp_min = 1.0
        self.alpha = 0.9
        self.eval_count = 0

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.apply_along_axis(func, 1, population)
        self.eval_count += self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Simulated Annealing Local Refinement
                if self.eval_count < self.budget:
                    temp = max(self.temp_min, self.temp_initial * (self.alpha ** (self.eval_count / self.budget)))
                    neighbor = trial + np.random.uniform(-0.5, 0.5, self.dim) * temp
                    neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                    neighbor_fitness = func(neighbor)
                    self.eval_count += 1

                    if neighbor_fitness < trial_fitness or np.random.rand() < np.exp((trial_fitness - neighbor_fitness) / temp):
                        population[i] = neighbor
                        fitness[i] = neighbor_fitness

                        if neighbor_fitness < best_fitness:
                            best_solution = neighbor
                            best_fitness = neighbor_fitness

        return best_solution, best_fitness