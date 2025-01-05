import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7
        self.population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def _mutate(self, idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        if np.random.rand() < 0.5:
            # Quantum-inspired mutation
            mutant += np.random.normal(0, 0.1, self.dim) * (self.best_solution - mutant)
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        trial = np.where(np.random.rand(self.dim) < self.crossover_probability, mutant, target)
        return trial

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i, self.lb, self.ub)
                trial = self._crossover(target, mutant)

                score = func(trial)
                eval_count += 1

                if score < func(target):
                    self.population[i] = trial
                    if score < self.best_score:
                        self.best_score = score
                        self.best_solution = trial

                # Adaptive mutation factor adjustment
                if score < self.best_score:
                    self.mutation_factor = min(1.0, self.mutation_factor * 1.2)
                else:
                    self.mutation_factor = max(0.1, self.mutation_factor * 0.9)

        return self.best_solution, self.best_score