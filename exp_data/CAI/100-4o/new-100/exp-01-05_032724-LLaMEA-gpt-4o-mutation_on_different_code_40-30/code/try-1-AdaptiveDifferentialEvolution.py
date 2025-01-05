import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.sigma = 0.5
        self.c_sigma = 0.3
        self.cov_matrix = np.eye(dim)
        self.F = 0.5 + np.random.rand(self.population_size) * 0.5
        self.CR = 0.5 + np.random.rand(self.population_size) * 0.5

    def mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.F[idx] * (self.population[b] - self.population[c])
        return np.clip(mutant, -5.0, 5.0)
    
    def crossover(self, target, mutant, idx):
        crossover_mask = np.random.rand(self.dim) < self.CR[idx]
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def update_covariance(self, step):
        self.cov_matrix = (1 - self.c_sigma) * self.cov_matrix + self.c_sigma * np.outer(step, step)
    
    def __call__(self, func):
        eval_count = 0
        self.fitness = np.array([func(x) for x in self.population])
        eval_count += self.population_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant, i)
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < self.fitness[i]:
                    step = trial - self.population[i]
                    self.update_covariance(step)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    self.F[i] = np.clip(self.F[i] + np.random.normal(0, 0.1), 0.1, 1.0)
                    self.CR[i] = np.clip(self.CR[i] + np.random.normal(0, 0.1), 0.1, 1.0)
            
            self.sigma *= np.exp((np.linalg.norm(self.x_opt) - np.linalg.norm(self.population.mean(axis=0)) - 1) / self.c_sigma)
        
        return self.f_opt, self.x_opt