import numpy as np

class QS_EDE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.elite_archive_size = 5
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.population = None
        self.scores = None
        self.elite_archive = []
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_elite_archive()

    def evaluate(self, solution):
        value = self.func(solution)
        self.evaluations += 1
        return value

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        F = np.random.uniform(self.F_min, self.F_max)
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            self.update_elite_archive()

    def quantum_inspired_mutation(self):
        for i in range(self.population_size):
            elite = self.elite_archive[np.random.randint(len(self.elite_archive))]
            quantum_superposition = np.random.uniform(-1, 1, self.dim)
            quantum_mutant = elite + quantum_superposition * np.abs(self.population[i] - elite)
            quantum_mutant = np.clip(quantum_mutant, self.func.bounds.lb, self.func.bounds.ub)
            self.select(i, quantum_mutant)

    def update_elite_archive(self):
        combined = list(zip(self.scores, self.population))
        combined.sort(key=lambda x: x[0])
        self.elite_archive = [ind for _, ind in combined[:self.elite_archive_size]]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                if self.evaluations >= self.budget:
                    break
            self.quantum_inspired_mutation()
        
        best_idx = np.argmin(self.scores)
        return {'solution': self.population[best_idx], 'fitness': self.scores[best_idx]}