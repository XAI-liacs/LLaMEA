import numpy as np

class EnhancedPeriodicInformedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.population = None
        self.best_solution = None
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.initialize_population()
        evaluations = 0
        F = self.initial_F
        CR = self.initial_CR

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                target = self.population[i]
                
                F = self.adapt_mutation_rate(F, func(target)) 
                CR = self.adapt_crossover_rate(CR, func(target)) 

                trial = self.mutate_and_crossover(i, F, CR)
                trial = self.enforce_periodicity_globally(trial)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness > func(target):  # Assume maximization
                    self.population[i] = trial
                    if self.best_solution is None or trial_fitness > func(self.best_solution):
                        self.best_solution = trial
                        F = np.clip(F + 0.05, 0.5, 1.0)

                if evaluations % (self.population_size // 2) == 0:  # Changed
                    self.enforce_diversity()  # Changed

        return self.best_solution

    def initialize_population(self):
        self.population = np.random.uniform(self.lb + (self.ub - self.lb) * 0.25, self.ub - (self.ub - self.lb) * 0.25, (self.population_size, self.dim)) 
        self.population += np.random.normal(scale=0.01, size=self.population.shape)

    def mutate_and_crossover(self, idx, F, CR):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        epsilon = np.random.uniform(0.9, 1.1)
        mutant = self.population[a] + F * (self.population[b] - self.population[c]) * epsilon
        mutant = np.clip(mutant, self.lb, self.ub)
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[idx])
        return trial

    def enforce_periodicity_globally(self, solution):
        period = 2 + np.random.randint(-1, 2)
        for start in range(0, self.dim, period):
            pattern = solution[start:start+period]
            for j in range(start, self.dim, period):
                if np.random.rand() < 0.8:
                    pattern_length = min(period, self.dim - j)
                    solution[j:j+pattern_length] = pattern[:pattern_length]
        return solution

    def adapt_mutation_rate(self, F, fitness):  
        adjustment = (fitness / (fitness + 1)) * 0.1  
        return min(1.0, F + (np.random.rand() - 0.5) * adjustment)  

    def adapt_crossover_rate(self, CR, fitness):  
        adjustment = (fitness / (fitness + 1)) * 0.1  
        return np.clip(CR + (np.random.rand() - 0.5) * adjustment, 0.5, 1.0)  

    def enforce_diversity(self):  # Changed
        if np.random.rand() < 0.5:  # Changed
            self.population += np.random.normal(scale=0.02, size=self.population.shape)  # Changed