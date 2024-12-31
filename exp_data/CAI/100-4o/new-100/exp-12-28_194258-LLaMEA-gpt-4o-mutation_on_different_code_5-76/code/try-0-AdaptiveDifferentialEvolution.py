import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget - self.pop_size):
            # Calculate diversity
            diversity = np.std(population, axis=0).mean()

            # Adaptive parameters
            F = np.clip(0.5 + 0.1 * np.tanh(diversity), 0.1, 0.9)
            CR = np.clip(0.9 - 0.1 * np.tanh(diversity), 0.1, 0.9)
            
            # Select three random vectors for mutation
            indices = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = population[indices]

            # Mutation and crossover
            mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, population[i % self.pop_size])

            # Evaluate trial vector
            f_trial = func(trial)
            if f_trial < fitness[i % self.pop_size]:
                population[i % self.pop_size] = trial
                fitness[i % self.pop_size] = f_trial

                # Update global best
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt