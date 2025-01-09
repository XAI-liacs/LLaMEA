import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # crossover probability
        self.adaptive_factor = 0.1
        self.local_search_prob = 0.2

    def mutate(self, idx, population):
        candidates = list(range(0, self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, individual, func):
        perturb = np.random.normal(0, 0.1, size=self.dim)
        candidate = np.clip(individual + perturb, -5.0, 5.0)
        return candidate if func(candidate) < func(individual) else individual

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)

                # Adaptive mutation factor
                if trial_fitness < fitness[i]:
                    self.F = min(1.0, self.F + self.adaptive_factor)
                else:
                    self.F = max(0.1, self.F - self.adaptive_factor)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Local search refinement
                if np.random.rand() < self.local_search_prob:
                    candidate = self.local_search(population[i], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt