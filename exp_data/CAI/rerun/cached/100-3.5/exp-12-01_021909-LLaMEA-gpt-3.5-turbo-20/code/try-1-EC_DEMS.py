import numpy as np

class EC_DEMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.num_strategies = 5
        self.strategy_probs = np.ones(self.num_strategies) / self.num_strategies
        self.strategy_factors = np.random.uniform(0.5, 1.0, size=(self.num_strategies, dim))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            selected_strategy = np.random.choice(self.num_strategies, p=self.strategy_probs)
            strategy_factor = self.strategy_factors[selected_strategy]
            mutant_population = population + np.random.normal(0, strategy_factor, size=(self.pop_size, self.dim))
            mutant_population = np.clip(mutant_population, -5.0, 5.0)
            mutant_fitness = np.array([func(individual) for individual in mutant_population])
            
            better_indices = np.where(mutant_fitness < fitness)[0]
            population[better_indices] = mutant_population[better_indices]
            fitness[better_indices] = mutant_fitness[better_indices]
            
            self.strategy_probs = 0.9 * self.strategy_probs + 0.1 * (better_indices.size / self.pop_size)
            self.strategy_probs /= np.sum(self.strategy_probs)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]