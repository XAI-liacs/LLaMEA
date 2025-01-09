import numpy as np

class HybridDESA:
    def __init__(self, budget, dim, F=0.8, Cr=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.Cr = Cr
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        
    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation (Differential Evolution)
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.Cr
                trial[crossover_points] = mutant[crossover_points]
                
                # Selection
                f_trial = func(trial)
                self.evaluations += 1
                if f_trial < func(self.population[i]):
                    self.population[i] = trial
                    if f_trial < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = f_trial
                
                # Simulated Annealing-like exploitation
                temperature = max(1.0, (self.budget - self.evaluations) / self.budget)
                new_solution = trial + np.random.normal(0, 0.1, self.dim) * temperature
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                f_new = func(new_solution)
                self.evaluations += 1
                if f_new < f_trial or np.exp((f_trial - f_new) / temperature) > np.random.rand():
                    trial = new_solution
                    f_trial = f_new
                    if f_trial < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = f_trial

        return self.best_solution