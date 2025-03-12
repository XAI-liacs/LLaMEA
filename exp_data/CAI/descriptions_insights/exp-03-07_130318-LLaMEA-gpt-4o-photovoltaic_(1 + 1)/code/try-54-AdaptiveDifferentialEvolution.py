import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.learning_rate = 0.01  # Added dynamic learning rate
        self.population = None
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, lb, ub):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        dynamic_factor = np.random.rand() * (1 - self.mutation_factor) + self.mutation_factor
        convergence_factor = 1 - np.std(self.population) / (np.ptp(self.population) + 1e-9)
        diversity_factor = np.random.rand()  # Introduce random diversity factor
        mutant = a + dynamic_factor * (b - c) + 0.1 * convergence_factor * (np.mean(self.population, axis=0) - a) + diversity_factor * (ub - lb) * np.random.normal(size=self.dim)
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        self.crossover_rate = 0.5 + 0.5 * (np.std(self.population) / np.mean(np.abs(self.population)))
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate, bounds):
        step_size = self.learning_rate * 0.005 * (bounds.ub - bounds.lb)  # Adjusted with learning rate
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        new_candidate = candidate + perturbation
        return np.clip(new_candidate, bounds.lb, bounds.ub)

    def update_population(self, new_population, bounds):
        diversity_threshold = 0.1 * self.dim
        if np.std(new_population) < diversity_threshold:
            self.initialize_population(bounds.lb, bounds.ub)
            print("Diversity too low, restarting population...")  # Add logging for restarts
        else:
            self.learning_rate *= 1.01  # Incrementally adjust learning rate

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
            
            self.update_population(new_population, bounds)  # Pass bounds for potential reset
            self.population = np.array(new_population)

        return best_solution