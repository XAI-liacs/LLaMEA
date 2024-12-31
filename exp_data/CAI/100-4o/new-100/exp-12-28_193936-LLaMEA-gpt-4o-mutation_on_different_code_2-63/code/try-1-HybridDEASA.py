import numpy as np

class HybridDEASA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.mutation_factor = 0.9  # Adjust mutation factor for enhanced exploration
        self.crossover_rate = 0.9
        self.temp = 1.0
        self.cooling_rate = 0.99
        self.iterations = self.budget // self.population_size

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.iterations):
            new_population = np.zeros_like(population)
            for j in range(self.population_size):
                # DE Mutation
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[j])

                # Evaluate Trial
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    new_population[j] = trial
                    fitness[j] = f_trial
                else:
                    new_population[j] = population[j]
            
            # ASA-inspired Adjustments
            for j in range(self.population_size):
                candidate = new_population[j] + np.random.normal(0, self.temp, self.dim)
                candidate = np.clip(candidate, bounds[0], bounds[1])
                f_candidate = func(candidate)
                
                if f_candidate < fitness[j] or np.exp((fitness[j] - f_candidate) / self.temp) > np.random.rand():
                    new_population[j] = candidate
                    fitness[j] = f_candidate
            
            population = new_population
            self.temp *= self.cooling_rate
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.f_opt:
                self.f_opt = fitness[min_idx]
                self.x_opt = population[min_idx]

        return self.f_opt, self.x_opt