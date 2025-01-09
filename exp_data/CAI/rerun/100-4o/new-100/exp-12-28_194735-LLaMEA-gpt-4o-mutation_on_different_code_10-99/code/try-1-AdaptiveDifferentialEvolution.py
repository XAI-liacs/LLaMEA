import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt, self.x_opt = min(zip(fitness, population), key=lambda x: x[0])
        
        budget_used = self.pop_size
        adaptive_rates = np.random.uniform(0.4, 0.9, size=(self.pop_size, 2))  # Initialize adaptive rates

        while budget_used < self.budget:
            new_population = population.copy()  # Prepare a new population for elitism strategy
            for i in range(self.pop_size):
                F, CR = adaptive_rates[i]  # Use individual adaptive rates

                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                budget_used += 1
                if f_trial < fitness[i]:
                    new_population[i] = trial  # Update new population instead of directly
                    fitness[i] = f_trial
                    adaptive_rates[i] = [F, CR]  # Successful rates are retained
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if budget_used >= self.budget:
                    break
            
            population = new_population  # Elitism strategy: retain the best solutions

            # Adaptive mechanism: update F and CR
            success_rate = np.mean([1 if fitness[i] < func(population[i]) else 0 for i in range(self.pop_size)])
            self.F = 0.4 + 0.3 * success_rate
            self.CR = 0.8 + 0.1 * success_rate

        return self.f_opt, self.x_opt