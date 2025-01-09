import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, max(10, dim * 5))  # Dynamic population size
        self.mutation_strategy = 'rand/1/bin'  # Initial mutation strategy
        self.CR = 0.9  # Crossover probability
        self.F = 0.8  # Differential weight

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Select mutation strategy based on remaining budget
                if eval_count / self.budget < 0.5:
                    self.mutation_strategy = 'rand/1/bin'
                else:
                    self.mutation_strategy = 'best/1/bin'

                # Mutation and crossover
                if self.mutation_strategy == 'rand/1/bin':
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant_vector = a + self.F * (b - c)
                else:  # 'best/1/bin'
                    best_idx = np.argmin(fitness)
                    indices = np.random.choice(self.population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant_vector = population[best_idx] + self.F * (a - b)

                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                # Dynamic adjustment of CR and F
                if eval_count % (self.budget // 10) == 0:
                    self.CR = np.random.uniform(0.5, 1.0)
                    self.F = np.random.uniform(0.5, 1.0)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]