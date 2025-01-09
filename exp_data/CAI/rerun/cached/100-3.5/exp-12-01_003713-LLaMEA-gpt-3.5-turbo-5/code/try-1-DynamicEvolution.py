import numpy as np

class DynamicEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5
        self.CR = 0.9
        self.NP = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def clipToBounds(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)

    def ensure_bounds(self, vec):
        vec_new = []
        for i in range(len(vec)):
            if vec[i] < self.lower_bound:
                vec_new.append(self.lower_bound)
            elif vec[i] > self.upper_bound:
                vec_new.append(self.upper_bound)
            else:
                vec_new.append(vec[i])
        return vec_new

    def mutation(self, x_r1, x_r2, x_r3):
        return self.clipToBounds(x_r1 + self.F * (x_r2 - x_r3))

    def crossover(self, x_current, x_mutant):
        j_rand = np.random.randint(0, self.dim)
        trial = [x_mutant[i] if (np.random.uniform(0, 1) < self.CR or i == j_rand) else x_current[i] for i in range(self.dim)]
        return self.ensure_bounds(trial)

    def __call__(self, func):
        population = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.NP)]
        fitness = [func(x) for x in population]
        
        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                random_indices = np.random.choice(range(self.NP), 3, replace=False)
                x_r1, x_r2, x_r3 = population[random_indices]
                x_mutant = self.mutation(x_r1, x_r2, x_r3)
                trial = self.crossover(population[i], x_mutant)
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best):
                best = population[best_idx]
        return best