import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.fitness = None
        self.success_rate = 0.5  # Added for feedback mechanism

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.Inf)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.Inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.f_opt:
                    self.f_opt = self.fitness[i]
                    self.x_opt = self.population[i]
    
    def mutate(self, idx, F, bounds):
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant, CR):
        mask = np.random.rand(self.dim) < CR
        if not np.any(mask):
            mask[np.random.randint(0, self.dim)] = True
        return np.where(mask, mutant, target)

    def adapt_parameters(self, diversity):
        CR = 0.9 * (1.0 - np.exp(-diversity))
        F = 0.8 + 0.2 * np.random.rand() * self.success_rate  # Modified with success_rate
        return F, CR

    def calculate_diversity(self):
        mean_vector = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_vector, axis=1))
        return diversity

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = self.pop_size
        self.evaluate_population(func)

        while eval_count < self.budget:
            diversity = self.calculate_diversity()
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                F, CR = self.adapt_parameters(diversity)
                mutant = self.mutate(i, F, bounds)
                trial = self.crossover(self.population[i], mutant, CR)
                
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.success_rate = min(1.0, self.success_rate + 0.05)  # Adjust success_rate
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                else:
                    self.success_rate = max(0.1, self.success_rate - 0.05)  # Adjust success_rate

        return self.f_opt, self.x_opt