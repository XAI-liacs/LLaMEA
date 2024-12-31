import numpy as np

class SA_DE_Optimizer:
    def __init__(self, budget=10000, dim=10, initial_temp=1.0, cooling_rate=0.99):
        self.budget = budget
        self.dim = dim
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population_size = max(4, self.dim)
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        population_fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(population_fitness)
        self.x_opt = population[np.argmin(population_fitness)]

        current_temp = self.initial_temp
        budget_left = self.budget - population_size

        F_base = 0.8  # Base mutation factor
        CR_base = 0.5  # Base crossover probability

        while budget_left > 0:
            new_population = []
            for i in range(population_size):
                if budget_left <= 0:
                    break

                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adjust mutation factor based on temperature
                F = F_base * (current_temp / self.initial_temp)
                mutant_vector = a + F * (b - c)
                mutant_vector = np.clip(mutant_vector, bounds[0], bounds[1])

                # Adjust crossover probability based on temperature
                CR = CR_base * (current_temp / self.initial_temp)
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                budget_left -= 1

                if (trial_fitness < population_fitness[i] or
                        np.random.rand() < np.exp((population_fitness[i] - trial_fitness) / current_temp)):
                    new_population.append(trial_vector)
                    population_fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial_vector

            population = np.array(new_population)
            current_temp *= self.cooling_rate

        return self.f_opt, self.x_opt