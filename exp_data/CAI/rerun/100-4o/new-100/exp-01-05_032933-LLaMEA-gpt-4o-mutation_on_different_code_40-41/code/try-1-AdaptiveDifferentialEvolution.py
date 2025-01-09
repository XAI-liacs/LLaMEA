import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size
        adaptive_pop_size = self.initial_pop_size

        while evaluations < self.budget:
            for i in range(adaptive_pop_size):
                # Mutation
                idxs = np.random.choice(adaptive_pop_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Check budget
                if evaluations >= self.budget:
                    break
            
            # Adapt F and CR dynamically
            self.F = np.random.uniform(0.4, 0.9)  # Adaptive F
            self.CR = np.random.uniform(0.1, 0.9)  # Adaptive CR

            # Dynamically adjust population size and maintain elitism
            adaptive_pop_size = int(self.initial_pop_size * (1 - evaluations / self.budget))
            elite_idx = np.argmin(fitness)
            population = np.vstack((population[:adaptive_pop_size], population[elite_idx:elite_idx+1]))
            fitness = np.append(fitness[:adaptive_pop_size], fitness[elite_idx])

        return self.f_opt, self.x_opt