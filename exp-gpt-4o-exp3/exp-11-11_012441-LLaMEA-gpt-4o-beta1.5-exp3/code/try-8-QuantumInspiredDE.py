import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9
        self.f = 0.8
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
        self.quantum_prob = 0.2
    
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

    def quantum_superposition(self):
        quantum_mutants = self.population + np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.quantum_prob
        quantum_mutants = np.clip(quantum_mutants, self.lower_bound, self.upper_bound)
        return quantum_mutants
    
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
        self.quantum_prob *= 0.95  # Gradually reduce quantum effect to emphasize exploitation
    
    def resize_population(self):
        if self.function_evals > 0.5 * self.budget:
            new_size = max(self.dim, int(self.population_size * 0.7))
            if new_size < self.population_size:
                indices = np.argsort(self.population_values)[:new_size]
                self.population = self.population[indices]
                self.population_values = self.population_values[indices]
                self.population_size = new_size
    
    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.resize_population()
            quantum_mutants = self.quantum_superposition()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                if np.random.rand() < self.quantum_prob:
                    trial = self.crossover(self.population[i], quantum_mutants[i])
                else:
                    trial = self.crossover(self.population[i], mutant)
                self.select(i, trial, func)
            generation += 1
        return self.best_solution