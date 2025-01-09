import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        func_values = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                
                f_trial = func(trial_vector)
                evaluations += 1
                if f_trial < func_values[i]:
                    population[i] = trial_vector
                    func_values[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial_vector
            
            # Adaptive adjustment of parameters
            self.mutation_factor = 0.5 + 0.5 * np.random.rand()
            self.crossover_rate = 0.5 + 0.5 * np.random.rand()
            
        return self.f_opt, self.x_opt