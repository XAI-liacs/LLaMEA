import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, population_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, population, func):
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            target_idx = np.random.choice(self.population_size)
            a, b, c = np.random.choice(self.population_size, 3, replace=False)
            mutant_vector = population[a] + self.F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover_mask, mutant_vector, population[target_idx])
            if func(trial_vector) < func(population[target_idx]):
                new_population[i] = trial_vector
            else:
                new_population[i] = population[target_idx]

        return new_population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for i in range(self.budget // self.population_size):
            population = self.evolve_population(population, func)
        
        f_values = np.array([func(x) for x in population])
        best_idx = np.argmin(f_values)
        if f_values[best_idx] < self.f_opt:
            self.f_opt = f_values[best_idx]
            self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt