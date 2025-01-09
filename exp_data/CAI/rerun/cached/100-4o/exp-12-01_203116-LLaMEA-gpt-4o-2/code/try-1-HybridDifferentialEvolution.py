import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), *self.bounds)
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

            # Adaptive Local Search
            local_search_radius = (self.bounds[1] - self.bounds[0]) * 0.05
            if eval_count + self.dim <= self.budget:
                for j in range(self.dim):
                    if eval_count >= self.budget:
                        break
                    perturbed = best.copy()
                    perturbed[j] += np.random.uniform(-local_search_radius, local_search_radius)
                    perturbed = np.clip(perturbed, *self.bounds)
                    perturbed_fitness = func(perturbed)
                    eval_count += 1
                    if perturbed_fitness < fitness[best_idx]:
                        best[j] = perturbed[j]
                        fitness[best_idx] = perturbed_fitness

        return best