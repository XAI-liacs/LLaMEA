import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5  # Differential weight
        self.CR = 0.9 # Crossover probability
        self.best_solution = None
        self.best_value = float('inf')
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            individual = lb + (ub - lb) * np.random.rand(self.dim)
            population.append(individual)
        return np.array(population)

    def mutate(self, target_idx, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, lb, ub)

    def quantum_mutation(self, candidate, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.5 * (1 - evaluation_ratio):
            phase = np.arccos(1 - 2 * np.random.rand(self.dim))
            direction = np.sign(np.random.rand(self.dim) - 0.5)
            quantum_movement = (ub - lb) * 0.05 * np.tan(phase) * direction
            candidate = candidate + quantum_movement
            candidate = np.clip(candidate, lb, ub)
        return candidate

    def crossover(self, target, mutant):
        trial = np.copy(target)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i, lb, ub)
                trial = self.crossover(self.population[i], mutant)
                trial = self.quantum_mutation(trial, lb, ub, evaluations / self.budget)
                
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial.copy()
                
                if trial_value < func(self.population[i]):
                    self.population[i] = trial
                
                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value