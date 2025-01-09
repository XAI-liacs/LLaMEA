import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(3 * np.log(dim))
        self.F = 0.8  # Differential weight
        self.CR = 0.9 # Crossover probability
        self.bounds = (-5.0, 5.0)
        self.evals = 0

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.evals += self.pop_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        while self.evals < self.budget:
            # Differential Evolution step
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                self.evals += 1

                # Greedy selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

            # Local search on the best individual found
            for _ in range(self.dim):
                if self.evals >= self.budget:
                    break
                candidate = best + np.random.normal(scale=0.05, size=self.dim)
                candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                f_candidate = func(candidate)
                self.evals += 1
                if f_candidate < fitness[best_idx]:
                    best = candidate
                    fitness[best_idx] = f_candidate

        return best