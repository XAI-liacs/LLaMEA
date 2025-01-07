import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.8  # Differential weight
        self.CR_max = 0.9
        self.CR_min = 0.1
        self.quantum_amplification_prob = 0.05
        self.population = None
        self.scores = None
        self.best_idx = None
        self.best_score = float('inf')
        self.best_solution = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        for i in range(self.population_size):
            score = func(self.population[i])
            if score < self.scores[i]:
                self.scores[i] = score
            if score < self.best_score:
                self.best_score = score
                self.best_solution = self.population[i]
        return self.scores

    def adaptive_crossover_rate(self, iteration, max_iterations):
        return self.CR_max - ((self.CR_max - self.CR_min) * iteration / max_iterations)

    def mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutation_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutation_vector

    def crossover(self, target, mutant, cr):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < cr:
                trial[i] = mutant[i]
        return trial

    def quantum_amplitude_amplification(self, candidate):
        probability = np.random.rand()
        if probability < self.quantum_amplification_prob:
            phase_flip = np.exp(1j * np.pi)
            candidate = np.real(candidate * phase_flip)
        return candidate

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            cr = self.adaptive_crossover_rate(iteration, max_iterations)
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant, cr)
                trial = self.quantum_amplitude_amplification(trial)
                trial_score = func(trial)
                func_calls += 1
                if trial_score < self.scores[i]:
                    new_population[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
                if func_calls >= self.budget:
                    break
            self.population = new_population
            iteration += 1
        
        return self.best_solution, self.best_score