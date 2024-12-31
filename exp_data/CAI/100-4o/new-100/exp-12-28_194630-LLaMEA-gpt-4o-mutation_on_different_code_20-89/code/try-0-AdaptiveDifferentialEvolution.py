import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 10 * dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Track the best solution
        min_index = np.argmin(fitness)
        self.f_opt = fitness[min_index]
        self.x_opt = population[min_index]
        
        # Run the optimization process
        num_evaluations = self.pop_size
        while num_evaluations < self.budget:
            F = np.random.normal(0.5, 0.3)  # adaptive scaling factor
            CR = np.random.uniform(0.1, 0.9)  # adaptive crossover rate
            
            for i in range(self.pop_size):
                # Select three random indices for mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation and recombination
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial vector
                f_trial = func(trial)
                num_evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    # Update the best found solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if num_evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt