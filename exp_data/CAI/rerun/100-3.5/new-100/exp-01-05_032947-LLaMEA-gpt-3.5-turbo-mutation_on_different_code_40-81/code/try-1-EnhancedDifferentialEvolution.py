import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = min(10 * dim, 200)
        self.F_min = 0.2
        self.F_max = 0.8
        self.CR = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.adaptation_rate = 0.1
        self.adaptation_counter = 0

    def __call__(self, func):
        for i in range(self.budget):
            self.adaptation_counter += 1
            if self.adaptation_counter % int(self.budget * self.adaptation_rate) == 0:
                self.population_size = max(int(self.population_size * 1.1), self.population_size)
                self.population = np.vstack([self.population, np.random.uniform(-5.0, 5.0, (int(self.population_size * 0.1), self.dim)])

            for j in range(self.population_size):
                idxs = np.arange(self.population_size)
                np.random.shuffle(idxs)
                a, b, c = self.population[np.random.choice(idxs[:3], 3, replace=False)]
                F = np.random.uniform(self.F_min, self.F_max)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                
                j_rand = np.random.randint(self.dim)
                trial = np.array([mutant[k] if np.random.rand() < self.CR or k == j_rand else self.population[j, k] for k in range(self.dim)])
                
                f_trial = func(trial)
                if f_trial < func(self.population[j]):
                    self.population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt