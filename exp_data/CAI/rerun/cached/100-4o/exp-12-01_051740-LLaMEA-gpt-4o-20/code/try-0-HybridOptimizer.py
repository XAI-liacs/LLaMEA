import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + 5 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.temperature = 1.0

    def __call__(self, func):
        evals = 0

        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i]
        
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Evaluation and Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

                # Adaptive Simulated Annealing
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                perturbed_solution = np.clip(self.best_solution + perturbation, self.lower_bound, self.upper_bound)
                perturbed_fitness = func(perturbed_solution)
                evals += 1
                if perturbed_fitness < self.best_fitness or \
                   np.exp((self.best_fitness - perturbed_fitness) / self.temperature) > np.random.rand():
                    self.best_solution = perturbed_solution
                    self.best_fitness = perturbed_fitness

            self.temperature *= 0.99  # Cooling schedule

        return self.best_solution, self.best_fitness