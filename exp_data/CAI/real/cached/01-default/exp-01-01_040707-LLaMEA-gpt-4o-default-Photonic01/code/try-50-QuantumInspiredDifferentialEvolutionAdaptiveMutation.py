import numpy as np

class QuantumInspiredDifferentialEvolutionAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.mutation_prob = 0.1  # Mutation probability
        self.position = None
        self.lb = None
        self.ub = None

    def initialize(self, bounds):
        self.lb, self.ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def evaluate(self, func):
        scores = np.array([func(ind) for ind in self.position])
        return scores

    def mutate(self, idx, best_idx):
        indices = np.random.permutation(self.population_size)
        indices = indices[indices != idx][:3]
        a, b, c = self.position[indices]
        best = self.position[best_idx]
        mutant = best + self.F * (a - b + c - self.position[idx])
        mutant = np.clip(mutant, self.lb, self.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def quantum_mutation(self, position):
        theta = np.random.rand(*position.shape) * np.pi / 4
        mutation = np.random.randn(*position.shape) * self.mutation_prob
        return position * np.cos(theta) + mutation * np.sin(theta)

    def select(self, target, trial, target_score, trial_score):
        if trial_score < target_score:
            return trial, trial_score
        return target, target_score

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        scores = self.evaluate(func)
        func_calls += self.population_size

        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        
        while func_calls < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i, best_idx)
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1

                self.position[i], scores[i] = self.select(self.position[i], trial, scores[i], trial_score)
                self.position[i] = self.quantum_mutation(self.position[i])

                if scores[i] < best_score:
                    best_score = scores[i]
                    best_idx = i

                if func_calls >= self.budget:
                    break

        return self.position[best_idx], best_score