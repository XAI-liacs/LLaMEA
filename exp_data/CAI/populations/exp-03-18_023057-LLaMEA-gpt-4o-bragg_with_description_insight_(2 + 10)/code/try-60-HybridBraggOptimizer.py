import numpy as np
from scipy.optimize import minimize

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8  # Changed line 1: Further increased mutation factor for exploration
        self.crossover_rate = 0.75  # Changed line 2: Adjusted crossover rate for diversity
        self.func_evals = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def _evaluate_population(self, func, population):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            if self.func_evals >= self.budget:
                break
            fitness[i] = func(population[i])
            self.func_evals += 1
        return fitness

    def _mutate(self, population, best_idx):
        indices = np.arange(self.population_size)
        for i in range(self.population_size):
            idxs = np.random.choice(indices[indices != i], 3, replace=False)
            target_vector = population[i]
            base_vector = population[best_idx]
            diff_vector = population[idxs[0]] - population[idxs[1]]
            mutant_vector = base_vector + np.random.uniform(0.5, 1.5) * diff_vector  # Changed line 3: Random scaling factor
            if np.random.rand() < 0.5:  # Encourage periodicity
                mutant_vector = 0.5 * mutant_vector + 0.5 * (target_vector + target_vector[::-1]) / 2  # Changed line 4
            yield mutant_vector

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        random_shift = np.random.randint(0, self.dim)  # This line remains
        mutant = np.roll(mutant, random_shift)  # This line remains
        return np.where(crossover_mask, mutant, target)

    def _local_optimization(self, func, position, bounds):
        result = minimize(func, position, bounds=bounds, method='L-BFGS-B')
        return result.x if result.success else position

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        population = self._initialize_population((func.bounds.lb, func.bounds.ub))
        fitness = self._evaluate_population(func, population)
        
        while self.func_evals < self.budget:
            best_idx = np.argmin(fitness)
            new_population = []

            for i, mutant in enumerate(self._mutate(population, best_idx)):
                if self.func_evals >= self.budget:
                    break
                trial_vector = self._crossover(population[i], mutant)
                trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial_vector)
                self.func_evals += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

            population = np.array(new_population)
            if self.func_evals < self.budget // 2:  # This line remains
                population[best_idx] = self._local_optimization(func, population[best_idx], bounds)
                fitness[best_idx] = func(population[best_idx])

        return population[np.argmin(fitness)]