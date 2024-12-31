import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = self.population[idxs]
                mutant = x0 + self.mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, -5, 5)
                
                # Crossover
                trial = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial[j] = mutant[j]

                # Evaluate and select
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Adaptive mechanisms
                self.adapt_rates()

        return self.f_opt, self.x_opt

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

    def adapt_rates(self):
        # Simple adaptation strategy: increase variation when stuck, decrease otherwise
        diversity = np.mean(np.std(self.population, axis=0))
        if diversity < 0.1:
            self.mutation_factor = min(1.0, self.mutation_factor + 0.1)
            self.crossover_rate = max(0.5, self.crossover_rate - 0.05)
        else:
            self.mutation_factor = max(0.5, self.mutation_factor - 0.05)
            self.crossover_rate = min(0.9, self.crossover_rate + 0.05)