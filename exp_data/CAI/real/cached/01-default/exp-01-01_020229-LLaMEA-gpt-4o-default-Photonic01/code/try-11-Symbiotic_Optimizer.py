import numpy as np

class Symbiotic_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutualism_factor = 1.5
        self.parasitism_factor = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize the population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        # Finding the best solution
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_value = fitness[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutualistic interaction
                partner_index = np.random.randint(0, self.population_size)
                if partner_index == i:
                    continue
                partner = population[partner_index]
                
                mutual_vector = (population[i] + partner) / 2
                mutual_vector = mutual_vector + self.mutualism_factor * (best_solution - mutual_vector)
                mutual_vector = np.clip(mutual_vector, lb, ub)
                
                mutual_value = func(mutual_vector)
                evaluations += 1

                if mutual_value < fitness[i]:
                    population[i] = mutual_vector
                    fitness[i] = mutual_value
                
                if mutual_value < best_value:
                    best_solution = mutual_vector
                    best_value = mutual_value
                
                if evaluations >= self.budget:
                    break
                
                # Parasitic interaction
                parasite_vector = np.random.uniform(lb, ub, self.dim)
                parasite_vector = parasite_vector + self.parasitism_factor * (best_solution - parasite_vector)
                parasite_vector = np.clip(parasite_vector, lb, ub)
                
                parasite_value = func(parasite_vector)
                evaluations += 1

                if parasite_value < fitness[partner_index]:
                    population[partner_index] = parasite_vector
                    fitness[partner_index] = parasite_value
                
                if parasite_value < best_value:
                    best_solution = parasite_vector
                    best_value = parasite_value
                
                if evaluations >= self.budget:
                    break
        
        return best_solution, best_value