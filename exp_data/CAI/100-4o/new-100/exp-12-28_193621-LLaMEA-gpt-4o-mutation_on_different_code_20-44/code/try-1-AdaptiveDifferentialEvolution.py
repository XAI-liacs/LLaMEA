import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, initial_population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = initial_population_size
        self.population_size = initial_population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.history = []
        self.local_search_prob = 0.2  # Added for local search

    def mutate(self, population):
        indices = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = population[indices]
        mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_parameters(self, success_rate):
        if success_rate > 0.2:
            self.F = min(1.0, self.F + 0.1)
            self.CR = max(0.1, self.CR - 0.1)
            self.population_size = min(self.initial_population_size * 2, self.population_size + 5)  # Dynamically adjust population size
        else:
            self.F = max(0.1, self.F - 0.1)
            self.CR = min(1.0, self.CR + 0.1)
            self.population_size = max(5, self.population_size - 5)

    def local_search(self, x, func):  # Added local search method
        perturbation = np.random.normal(0, 0.1, self.dim)
        x_new = np.clip(x + perturbation, -5.0, 5.0)
        return x_new if func(x_new) < func(x) else x

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            success_count = 0
            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:  # Perform local search with certain probability
                    trial = self.local_search(population[i], func)
                else:
                    mutant = self.mutate(population)
                    trial = self.crossover(population[i], mutant)
                
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_count += 1
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if evals >= self.budget:
                    break
            
            success_rate = success_count / self.population_size
            self.update_parameters(success_rate)

        return self.f_opt, self.x_opt