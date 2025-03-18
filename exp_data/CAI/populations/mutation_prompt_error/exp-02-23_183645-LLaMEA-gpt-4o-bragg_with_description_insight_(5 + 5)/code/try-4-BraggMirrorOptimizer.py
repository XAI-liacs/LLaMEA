import numpy as np

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        return pop

    def _evaluate_population(self, pop, func):
        fitness = np.apply_along_axis(func, 1, pop)
        self.evaluations += len(fitness)
        return fitness

    def _select_best(self, pop, fitness):
        idx = np.argmin(fitness)
        return pop[idx], fitness[idx]

    def _mutate(self, pop, idx):
        a, b, c = np.random.choice([i for i in range(self.population_size) if i != idx], 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search_periodicity(self, solution, func):
        # Encourage periodic solutions by averaging solutions to enforce a periodic pattern
        periodic_solution = solution.copy()
        period = self.dim // 2
        for i in range(period):
            avg_value = (solution[i] + solution[i + period]) / 2
            periodic_solution[i] = avg_value
            periodic_solution[i + period] = avg_value
        periodic_solution = np.clip(periodic_solution, func.bounds.lb, func.bounds.ub)
        fitness = func(periodic_solution)
        self.evaluations += 1
        return periodic_solution, fitness

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = self._evaluate_population(pop, func)

        self.best_solution, self.best_fitness = self._select_best(pop, fitness)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(pop, i)
                trial = self._crossover(pop[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution, self.best_fitness = trial, trial_fitness

                if self.evaluations >= self.budget:
                    break

            # Local search with periodicity enhancement
            local_best, local_best_fitness = self._local_search_periodicity(self.best_solution, func)
            if local_best_fitness < self.best_fitness:
                self.best_solution, self.best_fitness = local_best, local_best_fitness

        return self.best_solution