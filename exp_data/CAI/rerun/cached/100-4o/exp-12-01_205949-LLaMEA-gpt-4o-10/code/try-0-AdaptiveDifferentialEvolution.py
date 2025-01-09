import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim  # Initial population size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Differential weight
        self.scaled_budget = max(1, int(budget / 10))  # Budget scaling factor

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = np.random.choice(list(range(i)) + list(range(i+1, self.pop_size)), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.f * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.cr
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            population = new_population
            
            # Dynamic adjustment of differential weight
            self.f = 0.5 + (0.5 * eval_count / self.budget)

            # Adaptive population resizing
            if eval_count % self.scaled_budget == 0 and self.pop_size > self.dim:
                survivors = int(self.pop_size * 0.9)
                best_indices = np.argsort(fitness)[:survivors]
                population = population[best_indices]
                fitness = fitness[best_indices]
                self.pop_size = survivors

        best_idx = np.argmin(fitness)
        return population[best_idx]