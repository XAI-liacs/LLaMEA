import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 8 * dim  # Reduced population size
        self.mutation_factor = 0.7  # Adjusted mutation factor
        self.cross_prob = 0.85  # Adjusted crossover probability
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        best_idx = np.argmin(fitness)
        self.best_value = fitness[best_idx]
        self.best_solution = self.population[best_idx]
        self.evaluations += self.pop_size

        while self.evaluations < self.budget:
            next_gen = []

            for i in range(self.pop_size):
                if np.random.rand() < 0.7:
                    next_gen.append(self.variant_best(i, fitness, func))
                else:
                    next_gen.append(self.variant_rand1(i))

            self.population = np.array(next_gen)
            fitness = np.apply_along_axis(func, 1, self.population)
            current_best_idx = np.argmin(fitness)
            current_best_value = fitness[current_best_idx]
            
            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_solution = self.population[current_best_idx]
            
            # Elitist preservation
            self.population[np.random.randint(self.pop_size)] = self.best_solution
            self.evaluations += self.pop_size

        return self.best_solution

    def variant_best(self, index, fitness, func):
        best_idx = np.argmin(fitness)
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b = self.population[np.random.choice(idxs, 2, replace=False)]
        mutant = np.clip(self.population[best_idx] + self.mutation_factor * (a - b), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)

    def variant_rand1(self, index):
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
        return self.crossover(self.population[index], mutant)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cross_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial