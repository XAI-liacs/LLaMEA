import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Adjust pop_size dynamically based on the remaining budget
            self.pop_size = max(5, int(self.budget / (self.dim * 20)))  
            new_population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
            new_fitness = np.apply_along_axis(func, 1, new_population)
            population = np.vstack((population, new_population))
            fitness = np.append(fitness, new_fitness)
            
            for i in range(len(population)):
                indices = np.random.choice(len(population), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                if evaluations >= self.budget:
                    break

            self.mutation_factor *= 0.99  # Decay mutation factor to encourage exploitation
            self.crossover_rate *= 0.99  # Decay crossover rate slightly

        return self.f_opt, self.x_opt