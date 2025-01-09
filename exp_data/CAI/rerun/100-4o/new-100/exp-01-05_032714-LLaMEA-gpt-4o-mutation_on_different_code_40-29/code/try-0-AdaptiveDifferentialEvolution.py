import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent = self.pop_size

        while budget_spent < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = np.random.uniform(0.5, 1.0)
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                crossover_rate = 0.9
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
                
                # Apply a local search step by perturbing trial vector
                perturbation = np.random.normal(0, 0.1, self.dim)
                trial_local = np.clip(trial + perturbation, bounds[0], bounds[1])
                
                f_trial = func(trial)
                f_trial_local = func(trial_local)
                budget_spent += 2
                
                if f_trial_local < fitness[i]:
                    population[i] = trial_local
                    fitness[i] = f_trial_local
                elif f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

                if budget_spent >= self.budget:
                    break

        return self.f_opt, self.x_opt