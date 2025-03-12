import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.learning_rate = 0.01
        self.population = None
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, lb, ub):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        dynamic_factor = np.random.rand() * (1 - self.mutation_factor) + self.mutation_factor
        
        # Modification 1: Enhance dynamic mutation strategy
        scaling_factor = np.random.normal(0.5, 0.3)  # Normally distributed scaling factor
        mutant = a + scaling_factor * (b - c) + 0.1 * (np.mean(self.population, axis=0) - a)
        
        return np.clip(mutant, lb, ub)

    # Modification 2 & 3: Dynamic crossover rate and selection pressure
    def crossover(self, target, mutant):
        self.crossover_rate = 0.5 + 0.3 * np.tanh(np.std(self.population))  # Adjusted crossover rate
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate, bounds):
        # Modification 4: Intensified local search
        step_size = self.learning_rate * 0.005 * (bounds.ub - bounds.lb)
        perturbation = np.random.normal(0, step_size, self.dim)  # Normal distribution perturbation
        new_candidate = candidate + perturbation
        return np.clip(new_candidate, bounds.lb, bounds.ub)

    def adaptive_learning_rate(self):
        self.learning_rate = 0.01 * (1 - self.func_evals / self.budget)

    def update_population(self, new_population, bounds):
        diversity_threshold = 0.1 * self.dim
        if np.std(new_population) < diversity_threshold:
            self.initialize_population(bounds.lb, bounds.ub)
        else:
            self.learning_rate *= 1.01

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds.lb, bounds.ub)
        best_solution = None
        best_fitness = np.inf

        while self.func_evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                self.adaptive_learning_rate()
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
            
            self.update_population(new_population, bounds)
            self.population = np.array(new_population)

        return best_solution