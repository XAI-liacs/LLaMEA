import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            # Local search phase
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            for _ in range(int(self.pop_size * 0.1)):  # 10% of population size
                perturbation = np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
                new_fitness = func(new_solution)
                evaluations += 1

                if new_fitness < fitness[best_idx]:
                    population[best_idx] = new_solution
                    fitness[best_idx] = new_fitness

                if evaluations >= self.budget:
                    break

        return population[np.argmin(fitness)]