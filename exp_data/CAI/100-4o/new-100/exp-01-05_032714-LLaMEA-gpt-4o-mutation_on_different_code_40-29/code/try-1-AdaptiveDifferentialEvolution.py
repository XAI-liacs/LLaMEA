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
                F = np.random.uniform(0.6, 1.2)  # Adjusted mutation factor
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                crossover_rate = 0.8  # Slightly reduced crossover rate
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
                
                # Improved hybrid local search step
                perturbation_g = np.random.normal(0, 0.05, self.dim)
                perturbation_u = np.random.uniform(-0.05, 0.05, self.dim)
                trial_local = np.clip(trial + perturbation_g + perturbation_u, bounds[0], bounds[1])
                
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

            # Dynamic population adaptation
            if budget_spent < self.budget / 2 and self.pop_size < 50:
                new_individuals = np.random.uniform(bounds[0], bounds[1], (5, self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitness))
                self.pop_size += 5
                budget_spent += 5

        return self.f_opt, self.x_opt