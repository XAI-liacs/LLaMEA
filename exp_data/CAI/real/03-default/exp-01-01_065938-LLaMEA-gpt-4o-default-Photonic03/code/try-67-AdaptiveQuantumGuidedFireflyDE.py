import numpy as np

class AdaptiveQuantumGuidedFireflyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # Randomness scaling factor
        self.beta_min, self.beta_max = 0.2, 1.0  # Attraction coefficient scaling
        self.gamma = 1.0  # Light absorption coefficient
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.7, 1.0
        self.success_rates = [0.5, 0.5]
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.cos(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            successes = [0, 0]

            for i in range(self.population_size):
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                
                for j in range(self.population_size):
                    if fitness[i] > fitness[j]:
                        distance = np.linalg.norm(pop[i] - pop[j])
                        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * distance**2)
                        attraction = beta * (pop[j] - pop[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
                        pop[i] = pop[i] + attraction
                
                indices = np.random.choice(range(self.population_size), 2, replace=False)
                x1, x2 = pop[indices]
                
                mutant = pop[i] + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    next_pop[i] = trial
                    fitness[i] = trial_fitness
                    successes[0] += 1  # Track success
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            self.history.append(best_global)
            pop = next_pop

        return best_global