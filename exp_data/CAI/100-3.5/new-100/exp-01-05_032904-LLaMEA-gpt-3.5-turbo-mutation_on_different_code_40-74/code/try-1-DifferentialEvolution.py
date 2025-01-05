import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.scale_factor = 0.1

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                # Adaptive Scaling of Parameters
                self.F = np.clip(np.random.normal(self.F, self.scale_factor), 0.1, 0.9)
                self.CR = np.clip(np.random.normal(self.CR, self.scale_factor), 0.1, 0.9)
                
                # Mutation
                idxs = np.random.choice(pop_size, size=3, replace=False)
                mutant = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2])
                mutant = np.clip(mutant, bounds[0], bounds[1])
                
                # Crossover
                trial = np.copy(population[j])
                crossover_points = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]
                
                # Selection
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
        return self.f_opt, self.x_opt