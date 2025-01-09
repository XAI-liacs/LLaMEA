import numpy as np

class ScaleFreeAdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.evals = 0
        self.max_iter = budget // self.population_size

    def _generate_offspring(self, population, fitness, lb, ub):
        idxs = np.arange(self.population_size)
        np.random.shuffle(idxs)
        
        offspring = np.empty_like(population)
        for i in range(self.population_size):
            a, b, c = population[idxs[i]], population[idxs[(i+1)%self.population_size]], population[idxs[(i+2)%self.population_size]]
            mutant = np.clip(a + self.scale_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            offspring[i] = np.where(cross_points, mutant, population[i])
        return offspring

    def _local_search(self, ind, func, lb, ub):
        perturbation = np.random.uniform(-0.1, 0.1, size=self.dim) * (ub - lb)
        new_ind = np.clip(ind + perturbation, lb, ub)
        if func(new_ind) < func(ind):
            return new_ind
        return ind

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.evals = self.population_size
        
        best_idx = np.argmin(fitness)
        best_pos, best_fit = population[best_idx], fitness[best_idx]

        for iteration in range(self.max_iter):
            if self.evals >= self.budget:
                break

            offspring = self._generate_offspring(population, fitness, lb, ub)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.evals += self.population_size

            for i in range(self.population_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i] = offspring[i]
                    fitness[i] = offspring_fitness[i]

            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:
                    population[i] = self._local_search(population[i], func, lb, ub)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fit:
                best_pos, best_fit = population[best_idx], fitness[best_idx]

            self.scale_factor = 0.5 + 0.5 * np.sin(2 * np.pi * iteration / self.max_iter)
            self.crossover_rate = 0.9 - 0.4 * (iteration / self.max_iter)
            self.local_search_prob = 0.3 + 0.2 * np.cos(2 * np.pi * iteration / self.max_iter)
        
        return best_pos, best_fit