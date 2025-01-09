import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=100):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        pop_fitness = np.apply_along_axis(func, 1, pop)
        self.f_opt = np.min(pop_fitness)
        self.x_opt = pop[np.argmin(pop_fitness)]

        # Adaptive DE parameters
        F_base = 0.5
        CR_base = 0.9
        successful_F = []
        successful_CR = []
        
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                # Adaptive parameter control
                F = np.clip(F_base + np.random.normal(0, 0.1), 0, 2)
                CR = np.clip(CR_base + np.random.normal(0, 0.1), 0, 1)

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover_mask, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    successful_F.append(F)
                    successful_CR.append(CR)

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if evals >= self.budget:
                    break

            # Update adaptive parameters
            if successful_F:
                F_base = np.mean(successful_F)
                CR_base = np.mean(successful_CR)
                successful_F.clear()
                successful_CR.clear()

        return self.f_opt, self.x_opt