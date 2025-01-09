import numpy as np

class NovelOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0

    def __call__(self, func):
        def simulated_annealing_schedule(t):
            return max(0.01, np.exp(-0.005 * t))

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Mutation and recombination
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Adjust crossover probability over time
                dynamic_CR = self.CR * (1 - self.eval_count / self.budget)
                crossover = np.random.rand(self.dim) < dynamic_CR
                
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                self.eval_count += 1

                # Simulated annealing acceptance
                if trial_fitness < func(self.population[i]) or np.random.rand() < simulated_annealing_schedule(self.eval_count):
                    self.population[i] = trial

                # Update the best solution found
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

        return self.best_solution