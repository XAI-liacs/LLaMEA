import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, lb, ub):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        dynamic_factor = np.random.rand() * (1.2 - self.mutation_factor) + self.mutation_factor 
        if np.random.rand() > 0.5:
            mutant = a + dynamic_factor * (b - c)  # Alternating mutation strategies
        else:
            mutant = a + dynamic_factor * (np.mean(self.population, axis=0) - a)
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_rate = np.random.uniform(0.6, 0.9)  # Adaptive crossover rate
        crossover_mask = np.random.rand(self.dim) < crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate, bounds):
        step_size = 0.005 * (bounds.ub - bounds.lb)  
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        new_candidate = candidate + perturbation
        return np.clip(new_candidate, bounds.lb, bounds.ub)

    def update_population(self, new_population):
        diversity_threshold = 0.1 * self.dim
        if np.std(new_population) < diversity_threshold:
            self.initialize_population(bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds.lb, bounds.ub)
        best_solution = None
        best_fitness = np.inf

        while self.func_evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, bounds.lb, bounds.ub)
                trial = self.crossover(target, mutant)
                trial = self.local_search(trial, bounds)

                trial_fitness = func(trial)
                self.func_evals += 1

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

                if trial_fitness < func(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)

                if self.func_evals >= self.budget:
                    break
            
            self.update_population(new_population)
            self.population = np.array(new_population)

        return best_solution