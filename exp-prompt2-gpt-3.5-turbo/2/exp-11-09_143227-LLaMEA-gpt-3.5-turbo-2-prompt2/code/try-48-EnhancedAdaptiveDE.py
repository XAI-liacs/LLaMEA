import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)
        
        fitness_diversity = np.zeros(self.pop_size)
        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_fitness = np.min(fitness)
        
        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                
                diversity = np.mean(np.linalg.norm(population - target, axis=1))
                fitness_diversity[i] = diversity
                
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                    best_fitness = min(best_fitness, new_fitness)
                
                new_population.append(population[i])
                
            if _ % (self.budget // 10) == 0:
                best_idx = np.argmin(fitness)
                worst_idx = np.argmax(fitness_diversity)
                population[worst_idx] = population[best_idx]
                fitness[worst_idx] = fitness[best_idx]

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]