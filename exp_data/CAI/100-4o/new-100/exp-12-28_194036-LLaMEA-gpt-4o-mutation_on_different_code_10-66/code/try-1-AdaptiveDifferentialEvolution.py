import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        bounds = [(-5.0, 5.0)] * self.dim
        population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        if np.min(fitness) < self.f_opt:
            self.f_opt = fitness.min()
            self.x_opt = population[fitness.argmin()]
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                # Adapt mutation factor
                success_rate = np.count_nonzero(fitness < self.f_opt) / evaluations
                self.mutation_factor = 0.5 + success_rate * 0.3

                # Adjust crossover rate based on success
                diversity = np.std(population, axis=0).mean()
                self.crossover_rate = max(0.1, 0.9 - diversity * 0.5)

                if evaluations >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt