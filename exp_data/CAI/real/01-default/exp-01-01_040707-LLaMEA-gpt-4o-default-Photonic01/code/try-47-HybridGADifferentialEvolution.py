import numpy as np

class HybridGADifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_factor = 0.5
        self.adaptive_scale = 0.1
        self.position = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        for i in range(self.population_size):
            score = func(self.position[i])
            if score < self.scores[i]:
                self.scores[i] = score
            if score < self.best_score:
                self.best_score = score
                self.best_solution = self.position[i]
        return self.scores

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vec = self.position[a] + self.mutation_factor * (self.position[b] - self.position[c])
        return np.clip(mutant_vec, 0, 1)

    def crossover(self, parent, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, parent)
        return trial

    def adapt_parameters(self, iteration, max_iterations):
        self.crossover_rate = 0.7 + (0.9 - 0.7) * (iteration / max_iterations)
        self.mutation_factor = 0.5 + (0.9 - 0.5) * (1 - (iteration / max_iterations))

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        
        while func_calls < self.budget:
            self.adapt_parameters(iteration, max_iterations)
            scores = self.evaluate(func)
            func_calls += self.population_size

            new_population = []
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1

                if trial_score < self.scores[i]:
                    new_population.append(trial)
                    self.scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
                else:
                    new_population.append(self.position[i])

            self.position = np.array(new_population)
            iteration += 1

        return self.best_solution, self.best_score