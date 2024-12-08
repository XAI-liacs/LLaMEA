import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 12 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.85
        self.local_search_prob = 0.4
        self.adaptive_population_step = max(1, self.population_size // 5)

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant1 = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        d, e, f = pop[np.random.choice(indices, 3, replace=False)]
        mutant2 = np.clip(d + self.F * (e - f), self.lower_bound, self.upper_bound)
        return mutant1 if np.random.rand() > 0.5 else mutant2

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual, scale=1.0):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.05, 0.15) * scale
        new_individual = np.clip(individual + step_size * direction, self.lower_bound, self.upper_bound)
        return new_individual

    def _stochastic_ranking(self, population, fitness):
        indices = np.argsort(fitness)
        return population[indices], fitness[indices]

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

                if np.random.rand() < self.local_search_prob:
                    scale_factor = 1 - (evals / self.budget)
                    local_candidate = self._local_search(population[i], scale=scale_factor)
                    local_fitness = func(local_candidate)
                    evals += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

                if evals >= self.budget:
                    break

            if evals < self.budget:
                population, fitness = self._stochastic_ranking(population, fitness)
                if evals / self.budget > 0.5:
                    self.population_size = max(4, self.population_size - self.adaptive_population_step)
                    population = population[:self.population_size]
                    fitness = fitness[:self.population_size]

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]