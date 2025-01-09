import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.crossover_prob = 0.9
        self.mutation_factor = 0.8
        self.initial_temp = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        num_evaluations = 0

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations += self.population_size

        while num_evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Calculation of trial fitness
                trial_fitness = func(trial)
                num_evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / self.initial_temp)
                    if np.random.rand() < acceptance_prob:
                        new_population.append(trial)
                        fitness[i] = trial_fitness
                    else:
                        new_population.append(population[i])
            
            population = np.array(new_population)
            self.initial_temp *= self.cooling_rate

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]