import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.8  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.success_rate = 0.2  # For adaptive learning
        self.min_pop_size = 10
        self.max_pop_size = 40

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.pop_size
        
        while budget_used < self.budget:
            new_population = np.copy(population)
            successful_count = 0
            
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[indices]
                
                # Mutation
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    successful_count += 1
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                
                if budget_used >= self.budget:
                    break
            
            population = new_population

            # Adaptation of control parameters
            if successful_count / self.pop_size < self.success_rate:
                self.F = min(1.0, self.F * 1.2)
                self.CR = max(0.1, self.CR * 0.9)
                self.pop_size = max(self.min_pop_size, self.pop_size - 2)
            else:
                self.F = max(0.5, self.F * 0.9)
                self.CR = min(1.0, self.CR * 1.1)
                self.pop_size = min(self.max_pop_size, self.pop_size + 2)

        return self.f_opt, self.x_opt