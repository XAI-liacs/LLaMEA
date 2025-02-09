import numpy as np

class ChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.chaos_coefficient = 0.7

    def chaotic_sequence(self, size):
        x = np.random.rand()
        sequence = []
        for _ in range(size):
            x = self.chaos_coefficient * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            chaotic_factor = self.chaotic_sequence(1)[0]  # Update chaotic factor
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + chaotic_factor * self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                if evaluations >= self.budget:
                    break
            
            successful_trials = fitness < np.roll(fitness, 1)
            self.F = 0.5 + np.var(fitness) / np.mean(fitness)
            self.F = np.clip(self.F, 0.1, 1.0)
            self.CR = self.CR + 0.1 * (np.mean(successful_trials) - 0.5)
            self.CR = np.clip(self.CR, 0.1, 1.0)

        best_idx = np.argmin(fitness)
        return population[best_idx]