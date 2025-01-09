import numpy as np

class DEALS:
    def __init__(self, budget, dim, pop_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        self.eval_count += self.pop_size

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                trial_fitness = func(trial)
                self.eval_count += 1

                # Adaptive Local Search
                if trial_fitness < fitness[i]:
                    # Perform a local search around the trial solution
                    local_search_step = np.random.uniform(-0.1, 0.1, self.dim)
                    local_candidate = np.clip(trial + local_search_step, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_candidate)
                    self.eval_count += 1
                    
                    if local_fitness < trial_fitness:
                        trial = local_candidate
                        trial_fitness = local_fitness

                # Replacement
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best = trial

        return best