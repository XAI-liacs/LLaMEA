import numpy as np

class EvolutionaryGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        np.random.seed(42)
        population_size = 20
        step_size = 0.1
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                # Gradient-based local search
                grad = self.estimate_gradient(func, population[i], func(population[i]), step_size)
                new_solution = population[i] - step_size * grad
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = func(new_solution)
                evaluations += 1
                
                # Replace if new solution is better
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
            
            # Evolutionary step: crossover and mutation
            if evaluations < self.budget:
                parents_indices = np.random.choice(population_size, size=2, replace=False)
                parent1, parent2 = population[parents_indices]
                
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                
                # Mutation
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                child += mutation_vector
                child = np.clip(child, self.lower_bound, self.upper_bound)
                
                child_fitness = func(child)
                evaluations += 1
                
                # Replace worst in population if child is better
                worst_idx = np.argmax(fitness)
                if child_fitness < fitness[worst_idx]:
                    population[worst_idx] = child
                    fitness[worst_idx] = child_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def estimate_gradient(self, func, x, fx, step_size):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = np.array(x)
            x_plus[i] += step_size
            grad[i] = (func(x_plus) - fx) / step_size
        return grad