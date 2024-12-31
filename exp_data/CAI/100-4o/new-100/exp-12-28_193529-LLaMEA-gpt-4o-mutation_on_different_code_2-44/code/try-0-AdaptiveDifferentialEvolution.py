import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)  # Fixed bounds for BBOB problems
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Best solution initialization
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        # Parameters
        CR_base = 0.9
        F_base = 0.8
        success_rates = []
        evals = self.pop_size
        
        while evals < self.budget:
            CR = CR_base * (0.5 + 0.5 * np.random.rand())
            F = F_base * np.random.rand()
            
            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    success_rates.append(1)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    success_rates.append(0)
                
                if evals >= self.budget:
                    break
            
            # Adapt parameters
            if len(success_rates) > 50:
                success_rate = np.mean(success_rates[-50:])
                F_base = 0.9 * success_rate + 0.1 * (1 - success_rate)
                CR_base = 0.9 * (1 - success_rate) + 0.1 * success_rate

        return self.f_opt, self.x_opt