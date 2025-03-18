import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def quasi_oppositional_init(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        opp_pop = lb + ub - population
        return np.vstack((population, opp_pop))

    def mutate(self, target_idx, population):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def recombine(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, candidate, func, bounds):
        best = candidate
        for _ in range(5):
            candidate = candidate + np.random.normal(0, 0.01, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            if func(candidate) < func(best):
                best = candidate
        return best

    def __call__(self, func):
        self.pop_size = 20
        self.F = 0.8
        self.CR = 0.9
        func_evals = 0

        bounds = func.bounds
        population = self.quasi_oppositional_init(bounds.lb, bounds.ub)
        fitness = np.apply_along_axis(func, 1, population)
        func_evals += len(population)

        while func_evals < self.budget:
            new_population = []
            for i in range(self.pop_size):
                mutant = self.mutate(i, population)
                trial = self.recombine(population[i], mutant)
                trial = self.local_search(trial, func, bounds)  # Periodicity-enhancing local search
                trial_fitness = func(trial)
                func_evals += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if func_evals >= self.budget:
                    break

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]