import numpy as np

class HADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
    
    def __call__(self, func):
        rng = np.random.default_rng()
        pop = rng.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        evals = len(fitness)
        
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[rng.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = rng.random(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                # Local search: Randomly perturb the best solution found so far
                if evals < self.budget and rng.random() < 0.1:  # Introduce local search with probability
                    best_idx = np.argmin(fitness)
                    perturbed = pop[best_idx] + rng.normal(0, 0.1, self.dim)
                    perturbed = np.clip(perturbed, self.lower_bound, self.upper_bound)
                    f_perturbed = func(perturbed)
                    evals += 1
                    if f_perturbed < fitness[best_idx]:
                        pop[best_idx] = perturbed
                        fitness[best_idx] = f_perturbed

        return pop[np.argmin(fitness)]