import numpy as np

class CoevolutionaryHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.9

    def local_search(self, individual, func, bounds):
        step_size = 0.05 * (bounds[:, 1] - bounds[:, 0])
        best_local = individual
        best_local_value = func(individual)
        for _ in range(5):
            candidate = individual + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            candidate_value = func(candidate)
            if candidate_value < best_local_value:
                best_local = candidate
                best_local_value = candidate_value
        return best_local, best_local_value

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(pop)
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] 
                                 + self.c1 * r1 * (personal_best[i] - pop[i]) 
                                 + self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])

                trial_value = func(trial)
                eval_count += 1
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

            for i in range(self.population_size):
                improved_individual, improved_value = self.local_search(pop[i], func, bounds)
                if improved_value < personal_best_values[i]:
                    personal_best[i] = improved_individual
                    personal_best_values[i] = improved_value
                    if improved_value < global_best_value:
                        global_best = improved_individual
                        global_best_value = improved_value

                eval_count += 1
                if eval_count >= self.budget:
                    break

        return global_best