import numpy as np

class EfficientADELS_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5 * dim, 20)
        self.F = 0.7
        self.CR = 0.8

    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            idxs = np.random.randint(0, self.population_size, (self.population_size, 3))
            base_vecs = pop[idxs[:, 0]]
            diff_vectors = pop[idxs[:, 1]] - pop[idxs[:, 2]]
            trial_vectors = np.clip(base_vecs + self.F * diff_vectors, self.bounds[0], self.bounds[1])

            crossover_mask = (np.random.rand(self.population_size, self.dim) < self.CR)
            crossover_mask[np.arange(self.population_size), np.random.randint(0, self.dim, self.population_size)] = True
            offspring = np.where(crossover_mask, trial_vectors, pop)

            if np.random.rand() < 0.05:
                perturb_idx = np.random.randint(0, self.population_size)
                mutations = np.random.uniform(-0.05, 0.05, self.dim)
                offspring[perturb_idx] = np.clip(offspring[perturb_idx] + mutations, self.bounds[0], self.bounds[1])

            offspring_fitness = np.apply_along_axis(func, 1, offspring)
            evals += self.population_size

            improvement_mask = offspring_fitness < fitness
            pop[improvement_mask] = offspring[improvement_mask]
            fitness[improvement_mask] = offspring_fitness[improvement_mask]

            min_fitness_idx = np.argmin(offspring_fitness)
            if offspring_fitness[min_fitness_idx] < best_fitness:
                best_solution = offspring[min_fitness_idx]
                best_fitness = offspring_fitness[min_fitness_idx]

        return best_solution, best_fitness