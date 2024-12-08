import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.population_values[i] == float('inf'):
                self.population_values[i] = func(self.population[i])
                self.function_evals += 1
                if self.population_values[i] < self.best_value:
                    self.best_value = self.population_values[i]
                    self.best_solution = np.copy(self.population[i])
    
    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant
    
    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        self.function_evals += 1
        if trial_value < self.population_values[target_idx]:
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)
    
    def adapt_params(self, generation):
        self.cr = 0.9 - 0.5 * (generation / (self.budget / self.initial_population_size))
        self.f = 0.8 - 0.4 * (generation / (self.budget / self.initial_population_size))
    
    def resize_population(self):
        if self.function_evals > 0.5 * self.budget:
            new_size = max(self.dim, int(self.population_size * 0.7))
            if new_size < self.population_size:
                indices = np.argsort(self.population_values)[:new_size]
                self.population = self.population[indices]
                self.population_values = self.population_values[indices]
                self.population_size = new_size
    
    def local_search(self, solution, func):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        perturbed_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
        perturbed_value = func(perturbed_solution)
        self.function_evals += 1
        if perturbed_value < self.best_value:
            self.best_value = perturbed_value
            self.best_solution = perturbed_solution
        return perturbed_solution, perturbed_value
    
    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.resize_population()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                self.select(i, trial, func)
                if np.random.rand() < 0.1:  # Local search with a small probability
                    trial, _ = self.local_search(trial, func)
            generation += 1
        return self.best_solution