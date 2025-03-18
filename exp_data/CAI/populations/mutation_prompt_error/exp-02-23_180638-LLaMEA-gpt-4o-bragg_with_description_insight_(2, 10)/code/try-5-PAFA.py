import numpy as np

class PAFA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5
        self.beta_base = 1.0
        self.gamma = 1.0
        self.func_evals = 0
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.func_evals += self.population_size
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        rij = np.linalg.norm(pop[i] - pop[j])
                        beta = self.beta_base * np.exp(-self.gamma * rij ** 2)
                        pop[i] += beta * (pop[j] - pop[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
                        pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])
                        fitness[i] = func(pop[i])
                        self.func_evals += 1

                        if fitness[i] < fitness[best_idx]:
                            best_idx = i
                            best = pop[i]
                        
                        if self.func_evals >= self.budget:
                            break
            
            # Periodicity enforcement after each full population update
            for i in range(self.population_size):
                pop[i] = self.periodic_adjustment(pop[i], bounds)

        return best
    
    def periodic_adjustment(self, individual, bounds):
        # Encourage periodic patterns by adding a small sinusoidal adjustment
        period_factor = (bounds[:, 1] - bounds[:, 0]) / 4
        for d in range(self.dim):
            individual[d] += 0.1 * period_factor[d] * np.sin(2 * np.pi * individual[d] / period_factor[d])
        return np.clip(individual, bounds[:, 0], bounds[:, 1])