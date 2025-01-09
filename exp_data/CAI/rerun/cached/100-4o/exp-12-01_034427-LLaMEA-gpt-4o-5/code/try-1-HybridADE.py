import numpy as np

class HybridADE:
    def __init__(self, budget, dim, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0   # Upper bound of search space
        self.max_eval = budget
        self.eval_count = 0

    def __call__(self, func):
        # Initialize population
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]

        while self.eval_count < self.max_eval:
            for i in range(self.pop_size):
                if self.eval_count >= self.max_eval:
                    break

                # Mutation: create a mutant vector
                idxs = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                mutant = pop[idxs[0]] + self.F * (pop[idxs[1]] - pop[idxs[2]])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover: create a trial vector
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):  # ensure at least one crossover
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection: replace if trial is better
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

            # Adaptive restart if stagnant
            if self.eval_count < self.max_eval and self.eval_count % (self.pop_size * 5) == 0:
                sorted_indices = np.argsort(fitness)
                top_individuals = pop[sorted_indices[:self.pop_size // 5]]
                pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
                pop[:len(top_individuals)] = top_individuals
                fitness[:len(top_individuals)] = [func(ind) for ind in top_individuals]
                self.eval_count += len(top_individuals)

        return best_individual, best_fitness