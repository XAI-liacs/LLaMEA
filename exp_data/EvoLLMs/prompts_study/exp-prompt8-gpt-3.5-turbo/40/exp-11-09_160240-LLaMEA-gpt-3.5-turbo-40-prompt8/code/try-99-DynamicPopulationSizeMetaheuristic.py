class DynamicPopulationSizeMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        mutation_rates = np.full(pop_size, 0.5)
        crossover_probs = np.full(pop_size, 0.5)  # Introduce crossover probabilities
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                mutated_solution = ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                    scaling_factors[idx] *= 1.1
                    if np.random.uniform(0, 1) < crossover_probs[idx]:  # Utilize crossover probability
                        crossover_solution = ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim)
                        crossover_fitness = func(crossover_solution)
                        if crossover_fitness < fitness_values[idx]:
                            population[idx] = crossover_solution
                            fitness_values[idx] = crossover_fitness
                else:
                    scaling_factors[idx] *= 0.9
                    crossover_probs[idx] *= 0.95  # Adjust crossover probability
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
            
            if np.random.uniform(0, 1) < 0.1:  # Introduce dynamic population size mechanism
                pop_size = min(50, int(pop_size * 1.2))  # Increase population size dynamically
        
        return best_solution