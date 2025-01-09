import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 2*self.dim)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.cr = 0.9  # Crossover rate for DE
        self.f = 0.8   # Mutation factor for DE
    
    def adaptive_population_resize(self):
        if self.eval_count < self.budget // 2:
            self.population_size = max(10, int(self.population_size * 1.1))
        else:
            self.population_size = max(10, int(self.population_size * 0.9))
        self.population = self.population[:self.population_size]
        self.fitness = self.fitness[:self.population_size]

    def dynamic_mutation_factor(self):
        self.f = 0.5 + 0.3 * np.random.rand()

    def differential_evolution_step(self, func):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            f_trial = func(trial)
            self.eval_count += 1
            if f_trial < self.fitness[i]:
                new_population[i] = trial
                self.fitness[i] = f_trial
            if self.eval_count >= self.budget:
                break
        self.population = new_population

    def nelder_mead_step(self, func):
        sorted_indices = np.argsort(self.fitness)
        self.population = self.population[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
        
        centroid = np.mean(self.population[:-1], axis=0)
        reflected = np.clip(centroid + (centroid - self.population[-1]), self.bounds[0], self.bounds[1])
        f_reflected = func(reflected)
        self.eval_count += 1
        
        if f_reflected < self.fitness[0]:
            expanded = np.clip(centroid + 2 * (centroid - self.population[-1]), self.bounds[0], self.bounds[1])
            f_expanded = func(expanded)
            self.eval_count += 1
            if f_expanded < f_reflected:
                self.population[-1] = expanded
                self.fitness[-1] = f_expanded
            else:
                self.population[-1] = reflected
                self.fitness[-1] = f_reflected
        elif f_reflected < self.fitness[-2]:
            self.population[-1] = reflected
            self.fitness[-1] = f_reflected
        else:
            contracted = np.clip(centroid + 0.5 * (self.population[-1] - centroid), self.bounds[0], self.bounds[1])
            f_contracted = func(contracted)
            self.eval_count += 1
            if f_contracted < self.fitness[-1]:
                self.population[-1] = contracted
                self.fitness[-1] = f_contracted
            else:
                self.population[1:] = self.population[0] + 0.5 * (self.population[1:] - self.population[0])
                for i in range(1, self.population_size):
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1
                    if self.eval_count >= self.budget:
                        break

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1
            if self.eval_count >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            self.dynamic_mutation_factor()
            self.differential_evolution_step(func)
            self.nelder_mead_step(func)
            self.adaptive_population_resize()

        return self.population[np.argmin(self.fitness)]