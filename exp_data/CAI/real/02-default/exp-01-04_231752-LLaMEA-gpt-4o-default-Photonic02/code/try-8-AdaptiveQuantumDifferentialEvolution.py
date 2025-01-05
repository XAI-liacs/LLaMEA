import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.population = None
        self.scores = None
        self.best_vector = None
        self.best_score = np.inf
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.scores = np.full(self.population_size, np.inf)

    def _mutate(self, lb, ub, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
        
        # Quantum-inspired mutation
        quantum_factor = np.random.rand(self.dim) * (self.best_vector - self.population[idx])
        mutant_vector += quantum_factor
        
        return np.clip(mutant_vector, lb, ub)

    def _crossover(self, target, mutant):
        crossover_vector = np.copy(target)
        rand_idx = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == rand_idx:
                crossover_vector[j] = mutant[j]
        return crossover_vector

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                target_vector = self.population[i]
                mutant_vector = self._mutate(self.lb, self.ub, i)
                trial_vector = self._crossover(target_vector, mutant_vector)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial_vector
                    self.scores[i] = trial_score

                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_vector = trial_vector

        return self.best_vector, self.best_score