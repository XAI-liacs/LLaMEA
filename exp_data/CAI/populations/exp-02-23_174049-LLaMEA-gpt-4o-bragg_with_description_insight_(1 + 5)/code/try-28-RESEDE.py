import numpy as np
from scipy.optimize import minimize

class RESEDE:
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

    def adaptive_crossover(self, target, mutant, CR=0.9):
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        trial = (trial + np.roll(trial, 1)) / 2  # Promote periodicity
        return trial

    def fourier_constraint(self, candidate):
        transformed = np.fft.fft(candidate)
        filtered = np.where(np.abs(transformed) < np.max(np.abs(transformed)) * 0.2, 0, transformed)
        return np.real(np.fft.ifft(filtered))

    def stochastic_local_search(self, candidate, func, bounds):
        perturbation = np.random.normal(0, 0.01, size=self.dim)
        perturbed_candidate = candidate + perturbation
        res = minimize(func, perturbed_candidate, bounds=bounds, method='L-BFGS-B')
        return res.x, res.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        scaled_lb = np.zeros(self.dim)
        scaled_ub = np.ones(self.dim)

        pop_size = 20
        pop = self.quasi_oppositional_initialization(scaled_lb, scaled_ub, pop_size)
        pop = lb + pop * (ub - lb)

        fitness = np.array([func(ind) for ind in pop])
        self.history.extend(fitness)
        budget_spent = len(pop)

        while budget_spent < self.budget:
            for i in range(pop_size):
                if budget_spent >= self.budget:
                    break

                mutant = self.mutate(i, pop, F=np.random.uniform(0.5, 1.0))
                trial = self.adaptive_crossover(pop[i], mutant, CR=np.random.uniform(0.7, 1.0))

                trial = np.clip(trial, lb, ub)
                trial = self.fourier_constraint(trial)  # Apply Fourier constraint

                trial_fitness = func(trial)
                budget_spent += 1
                self.history.append(trial_fitness)

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

            if budget_spent < self.budget:
                best_idx = np.argmin(fitness)
                best_candidate, local_fitness = self.stochastic_local_search(pop[best_idx], func, list(zip(lb, ub)))

                if local_fitness < fitness[best_idx]:
                    pop[best_idx] = best_candidate
                    fitness[best_idx] = local_fitness
                    budget_spent += 1
                    self.history.append(local_fitness)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]