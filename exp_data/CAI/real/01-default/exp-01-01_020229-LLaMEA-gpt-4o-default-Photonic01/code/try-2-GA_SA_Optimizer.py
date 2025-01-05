import numpy as np

class GA_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Initial population size
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.initial_temperature = 1000.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        evaluations = self.population_size
        temperature = self.initial_temperature
        
        while evaluations < self.budget:
            # Genetic Algorithm Step
            new_population = []
            for _ in range(self.population_size // 2):
                parents = np.random.choice(self.population_size, 2, replace=False, p=fitness/fitness.sum())
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate([population[parents[0], :crossover_point], 
                                             population[parents[1], crossover_point:]])
                    child2 = np.concatenate([population[parents[1], :crossover_point], 
                                             population[parents[0], crossover_point:]])
                else:
                    child1, child2 = population[parents]
                    
                new_population.extend([child1, child2])
            
            # Mutation
            for i in range(len(new_population)):
                if np.random.rand() < self.mutation_rate:
                    mutation_index = np.random.randint(self.dim)
                    new_population[i][mutation_index] = np.random.uniform(lb[mutation_index], ub[mutation_index])
            
            # Evaluate new population
            new_population = np.array(new_population)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)
            
            # Combine and select the best
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]
            
            current_index = np.argmin(fitness)
            current_solution = population[current_index]
            current_fitness = fitness[current_index]
            
            # Simulated Annealing Step
            if np.random.rand() < np.exp((best_fitness - current_fitness) / temperature):
                best_solution = current_solution
                best_fitness = current_fitness

            # Cooling
            temperature *= self.cooling_rate

        return best_solution, best_fitness