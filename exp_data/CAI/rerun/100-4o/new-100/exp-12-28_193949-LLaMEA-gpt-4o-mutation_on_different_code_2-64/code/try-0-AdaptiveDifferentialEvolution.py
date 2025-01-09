import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * self.dim
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        f_values = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(f_values)
        self.f_opt = f_values[best_idx]
        self.x_opt = pop[best_idx].copy()
        evals = self.population_size

        F = 0.5
        CR = 0.9

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                    
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, pop[i])

                # Evaluation
                f_trial = func(trial)
                evals += 1

                # Selection
                if f_trial < f_values[i]:
                    pop[i] = trial
                    f_values[i] = f_trial

                    # Update the best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                        
            # Adaptive strategy: introduce random samples every few generations
            if evals + self.population_size < self.budget:
                random_samples = np.random.uniform(self.bounds[0], self.bounds[1], (self.dim, self.dim))
                random_values = np.array([func(ind) for ind in random_samples])
                evals += self.dim
                
                # Replace worst individuals in the population with random samples
                worst_indices = np.argsort(f_values)[-self.dim:]
                pop[worst_indices] = random_samples
                f_values[worst_indices] = random_values

                # Update best solution from new random samples
                best_random_idx = np.argmin(random_values)
                if random_values[best_random_idx] < self.f_opt:
                    self.f_opt = random_values[best_random_idx]
                    self.x_opt = random_samples[best_random_idx].copy()
                    
        return self.f_opt, self.x_opt