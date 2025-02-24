import numpy as np
from scipy.optimize import minimize

class ESEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []

    def quasi_oppositional_initialization(self, lb, ub, size):
        pop = np.random.uniform(low=lb, high=ub, size=(size, self.dim))
        opp_pop = lb + ub - pop
        combined_pop = np.vstack((pop, opp_pop))
        return combined_pop

    def mutate(self, target_idx, pop, F=0.5):
        idxs = [idx for idx in range(len(pop)) if idx != target_idx]
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant, CR=0.9):
        if len(self.history) > 2 and self.history[-1] > self.history[-2]:  # Adaptive CR tuning
            CR *= 0.95
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)

        # Promote periodicity by averaging adjacent elements
        trial = (trial + np.roll(trial, 1)) / 2
        return trial

    def local_search(self, candidate, func, bounds):
        res = minimize(func, candidate, bounds=bounds, method='L-BFGS-B')
        return res.x, res.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        scaled_lb = np.zeros(self.dim)
        scaled_ub = np.ones(self.dim)

        pop_size = 20
        pop = self.quasi_oppositional_initialization(scaled_lb, scaled_ub, pop_size)
        pop = lb + pop * (ub - lb)  # Scale population to actual bounds

        fitness = np.array([func(ind) for ind in pop])
        self.history.extend(fitness)
        budget_spent = len(pop)

        while budget_spent < self.budget:
            for i in range(pop_size):
                if budget_spent >= self.budget:
                    break

                mutant = self.mutate(i, pop)
                trial = self.crossover(pop[i], mutant)

                # Ensure trial is within bounds
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                budget_spent += 1
                self.history.append(trial_fitness)

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

            # Local search on the best candidate
            if budget_spent < self.budget:
                best_idx = np.argmin(fitness)
                best_candidate, local_fitness = self.local_search(pop[best_idx], func, list(zip(lb, ub)))

                if local_fitness < fitness[best_idx]:
                    pop[best_idx] = best_candidate
                    fitness[best_idx] = local_fitness
                    budget_spent += 1
                    self.history.append(local_fitness)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]