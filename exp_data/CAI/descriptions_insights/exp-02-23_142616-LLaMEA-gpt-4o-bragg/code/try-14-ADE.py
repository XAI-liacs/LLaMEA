import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.F_min, self.F_max = 0.5, 1.0
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                CR = self.CR_min + (self.CR_max - self.CR_min) * (self.budget - evaluations) / self.budget

                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                if evaluations >= self.budget:
                    break
        
        best_idx = scores.argmin()
        return population[best_idx], scores[best_idx]