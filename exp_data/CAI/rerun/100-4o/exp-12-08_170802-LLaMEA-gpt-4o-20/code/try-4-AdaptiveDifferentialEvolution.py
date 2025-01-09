import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size
        conv_rate = []

        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    fitness_improvement = fitness[i] - f_trial
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        conv_rate.append(fitness_improvement)

                # Self-adaptive mechanism
                if evals % (self.pop_size * 10) == 0:
                    f_mean = fitness.mean()
                    recent_conv = np.mean(conv_rate[-self.pop_size:])
                    if self.f_opt < f_mean:
                        self.F = min(self.F + 0.1, 1.0)
                        self.CR = max(self.CR - 0.1, 0.1)
                        if recent_conv < 0.01:  # Adjust population size based on convergence speed
                            self.pop_size = max(10, self.pop_size // 2)
                            population = population[:self.pop_size]
                            fitness = fitness[:self.pop_size]
                    else:
                        self.F = max(self.F - 0.1, 0.1)
                        self.CR = min(self.CR + 0.1, 1.0)
                        if recent_conv > 0.05:
                            self.pop_size = min(self.pop_size * 2, self.budget // self.dim)
                            new_individuals = np.random.uniform(bounds[0], bounds[1], (self.pop_size - len(population), self.dim))
                            population = np.vstack((population, new_individuals))
                            fitness = np.append(fitness, [func(ind) for ind in new_individuals])
                            evals += len(new_individuals)

        return self.f_opt, self.x_opt