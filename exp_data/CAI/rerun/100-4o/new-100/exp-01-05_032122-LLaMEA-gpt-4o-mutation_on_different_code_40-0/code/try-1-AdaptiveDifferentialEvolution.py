import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        elite_fraction = 0.2
        elite_size = int(self.pop_size * elite_fraction)
        
        while evaluations < self.budget:
            new_population = []
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), func.bounds.lb, func.bounds.ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                new_population.append(trial)
            
            population = np.array(new_population)
            top_elite = population[np.argsort(fitness)[:elite_size]]
            for elite in top_elite:
                local_mutant = elite + np.random.normal(0, 0.1, size=self.dim)
                local_mutant = np.clip(local_mutant, func.bounds.lb, func.bounds.ub)
                f_local = func(local_mutant)
                evaluations += 1
                if f_local < self.f_opt:
                    self.f_opt = f_local
                    self.x_opt = local_mutant

            if evaluations % (self.pop_size * 10) == 0:
                mean_fitness = np.mean(fitness)
                if np.std(fitness) < 0.1 * mean_fitness:
                    self.F = np.clip(self.F + np.random.normal(0, 0.1), 0.1, 0.9)
                    self.CR = np.clip(self.CR + np.random.normal(0, 0.1), 0.1, 0.9)

        return self.f_opt, self.x_opt