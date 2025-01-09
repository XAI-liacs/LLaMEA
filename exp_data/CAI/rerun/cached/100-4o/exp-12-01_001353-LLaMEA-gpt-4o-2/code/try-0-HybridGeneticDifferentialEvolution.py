import numpy as np

class HybridGeneticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.inf * np.ones(self.pop_size)
        self.used_budget = 0

    def select_parents(self):
        selected_indices = np.random.choice(self.pop_size, 2, replace=False)
        return self.population[selected_indices]

    def differential_evolution_step(self, target_idx):
        indices = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
        return mutant
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget += self.pop_size

        while self.used_budget < self.budget:
            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self.differential_evolution_step(i)
                offspring = self.crossover(target, mutant)
                
                offspring_fitness = func(offspring)
                self.used_budget += 1

                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness

                if self.used_budget >= self.budget:
                    break
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]