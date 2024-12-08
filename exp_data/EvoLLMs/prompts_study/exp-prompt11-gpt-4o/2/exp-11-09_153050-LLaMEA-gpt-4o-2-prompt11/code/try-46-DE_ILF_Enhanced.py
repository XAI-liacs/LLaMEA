import numpy as np

class DE_ILF_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(9 * dim)  # Reduced population size for faster convergence
        self.initial_scaling_factor = 0.8
        self.initial_crossover_prob = 0.85  # Slightly reduced initial crossover probability

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        for iteration in range(self.budget - self.population_size):
            scaling_factor = self.initial_scaling_factor * (1 - iteration / self.budget)
            crossover_prob = self.initial_crossover_prob * (1 - iteration / (2 * self.budget))  # Adaptive crossover rate
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + scaling_factor * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                if np.random.rand() < 0.5:
                    levy_step = self.levy_flight(self.dim)
                    trial = np.clip(trial + levy_step, self.lower_bound, self.upper_bound)
                
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]