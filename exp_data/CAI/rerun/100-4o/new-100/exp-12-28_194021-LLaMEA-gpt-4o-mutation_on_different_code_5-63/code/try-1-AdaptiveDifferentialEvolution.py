import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_base = 0.5
        self.CR_base = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.population_size
        
        stagnation_counter = 0  # Line 1 changed: add a stagnation counter
        while evaluations < self.budget:
            F, CR = self._adapt_parameters(evaluations / self.budget)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    stagnation_counter = 0  # Reset on improvement
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    stagnation_counter += 1  # Increment on no improvement
                
                if evaluations >= self.budget or stagnation_counter >= 100:  # Line 2 changed: trigger diversity mechanism
                    # Reinitialize a part of the population if stagnated
                    reinit_indices = np.random.choice(self.population_size, self.population_size // 5, replace=False)
                    pop[reinit_indices] = np.random.uniform(lb, ub, (len(reinit_indices), self.dim))
                    fitness[reinit_indices] = np.array([func(ind) for ind in pop[reinit_indices]])
                    stagnation_counter = 0

        return self.f_opt, self.x_opt

    def _adapt_parameters(self, progress):
        F = self.F_base + 0.1 * (0.5 - np.random.rand()) * np.sqrt(progress)
        CR = self.CR_base - 0.1 * np.random.rand() * np.sqrt(progress)
        return np.clip(F, 0.1, 1.0), np.clip(CR, 0.1, 1.0)