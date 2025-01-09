import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, Cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.Cr = Cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for eval_count in range(self.population_size, self.budget):
            idx = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = population[idx]
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            trial = np.copy(population[eval_count % self.population_size])
            
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.Cr or j == j_rand:
                    trial[j] = mutant[j]
            
            f_trial = func(trial)
            target_idx = eval_count % self.population_size
            
            if f_trial < fitness[target_idx]:
                population[target_idx] = trial
                fitness[target_idx] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    # Local search around the best solution
                    local_perturb = np.random.uniform(-0.1, 0.1, self.dim)
                    local_trial = np.clip(self.x_opt + local_perturb, func.bounds.lb, func.bounds.ub)
                    f_local = func(local_trial)
                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = local_trial

        return self.f_opt, self.x_opt