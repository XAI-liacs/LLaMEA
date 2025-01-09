import numpy as np

class DynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR = 0.9
        self.M_F = 0.5
        self.M_CR = 0.5
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                F = np.random.normal(self.M_F, 0.1)
                F = np.clip(F, self.F_min, self.F_max)
                CR = np.random.normal(self.M_CR, 0.1)
                CR = np.clip(CR, 0, 1)
                
                trial_vector = self.pop[a] + F * (self.pop[b] - self.pop[c])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial_vector[j] = np.clip(trial_vector[j], self.lower_bound, self.upper_bound)
                    else:
                        trial_vector[j] = self.pop[i, j]
                
                if func(trial_vector) < func(self.pop[i]):
                    self.pop[i] = trial_vector
                    self.M_F = 0.5 + 0.2 * (1 - (0.9 * _ / self.budget))
                    self.M_CR = 0.5 + 0.1 * (1 - (0.9 * _ / self.budget))
        
        best_solution = self.pop[np.argmin([func(individual) for individual in self.pop])]
        return best_solution