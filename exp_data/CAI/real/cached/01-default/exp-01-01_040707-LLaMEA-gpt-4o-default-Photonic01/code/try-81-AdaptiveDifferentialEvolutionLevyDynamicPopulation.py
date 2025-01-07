import numpy as np

class AdaptiveDifferentialEvolutionLevyDynamicPopulation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 20
        self.population_size = self.base_population_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.alpha = 1.5  # Lévy flight exponent
        self.position = None
        self.scores = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))
    
    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.scores[i]:
                self.scores[i] = scores[i]
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_position = self.position[i]
        return scores
    
    def levy_flight(self):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / self.alpha)
        return step
    
    def mutate_and_crossover(self, idx):
        a, b, c = np.random.choice(self.population_size, 3, replace=False)
        mutant = self.position[a] + self.F * (self.position[b] - self.position[c])
        trial = np.copy(self.position[idx])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial
    
    def adjust_population(self, iteration, max_iterations):
        self.population_size = self.base_population_size + int((self.budget // self.base_population_size) * (iteration / max_iterations))
        self.position = np.resize(self.position, (self.population_size, self.dim))
        self.scores = np.resize(self.scores, self.population_size)
    
    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.base_population_size
        iteration = 0
        
        while func_calls < self.budget:
            new_position = np.copy(self.position)
            for i in range(self.population_size):
                trial = self.mutate_and_crossover(i)
                trial += self.levy_flight()
                trial_score = func(trial)
                func_calls += 1
                if trial_score < self.scores[i]:
                    new_position[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = trial
            self.position = new_position
            self.adjust_population(iteration, max_iterations)
            iteration += 1
        
        return self.best_position, self.best_score