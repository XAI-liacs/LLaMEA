import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, min(100, budget // 5))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.jump_prob = 0.1  # Probability to apply quantum jump

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fit = func(self.population[i])
            if fit < self.fitness[i]:
                self.fitness[i] = fit
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = self.population[i]

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutated = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutated

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adapt_parameters(self):
        self.F = np.random.uniform(0.5, 1.0)
        self.CR = np.random.uniform(0.1, 0.9)

    def quantum_jump(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.jump_prob:
                jump_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], jump_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                
                trial_fitness = func(trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            self.adapt_parameters()
            self.quantum_jump(lb, ub)

        return self.best_solution, self.best_fitness