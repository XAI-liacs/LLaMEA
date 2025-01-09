import numpy as np

class HFA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0
        self.delta = 0.97
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        beta = self.beta_min + (1.0 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(pop[i] - pop[j])**2)
                        step = beta * (pop[j] - pop[i]) + self.alpha * np.random.uniform(self.lb, self.ub, self.dim)
                        new_ind = np.clip(pop[i] + step, self.lb, self.ub)
                        
                        if func(new_ind) < fitness[i]:
                            pop[i] = new_ind
                            fitness[i] = func(new_ind)
            
            best_idx = np.argmin(fitness)
            best_solution = pop[best_idx]
            
            for i in range(self.population_size):
                mutant = pop[np.random.choice(self.population_size, 3, replace=False)]
                trial = pop[i] + self.delta * (mutant[0] - pop[i]) + self.delta * (mutant[1] - mutant[2])
                trial_fitness = func(trial)
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
            
        return best_solution