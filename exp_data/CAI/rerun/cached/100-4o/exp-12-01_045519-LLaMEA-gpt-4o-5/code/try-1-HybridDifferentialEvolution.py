import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.num_evals = 0
    
    def __call__(self, func):
        best_score = np.inf
        best_sol = None

        while self.num_evals < self.budget:
            new_pop = np.zeros_like(self.pop)
            for i in range(self.population_size):
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = self.pop[idxs]

                v = x1 + self.F * (x2 - x3)
                v = np.clip(v, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, v, self.pop[i])

                score_trial = func(trial)
                if score_trial < best_score:
                    best_score = score_trial
                    best_sol = trial

                if score_trial < func(self.pop[i]):
                    new_pop[i] = trial
                else:
                    new_pop[i] = self.pop[i]

                self.num_evals += 1
                if self.num_evals >= self.budget:
                    break

            self.pop = new_pop
            # Adaptive control strategies
            self.F = np.clip(self.F + 0.1 * np.random.normal(), 0.1, 1.0)
            self.CR = np.clip(self.CR + 0.1 * np.random.normal(), 0.1, 1.0)
            
            # Dynamic population resizing
            if np.std([func(ind) for ind in self.pop]) < 1e-5:
                self.population_size = max(10, self.population_size // 2)  # Reduce population size

        return best_sol