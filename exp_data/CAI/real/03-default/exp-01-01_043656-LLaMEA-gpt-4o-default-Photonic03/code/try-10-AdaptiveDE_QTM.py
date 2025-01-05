import numpy as np

class AdaptiveDE_QTM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_pop_size = min(100, 10 * dim)
        self.pop_size = self.base_pop_size
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.adaptive_threshold = 0.2

    def quantum_tunneling(self, position, best):
        delta = np.random.rand(self.dim)
        factor = np.random.normal(0, 1, self.dim)
        return position + factor * (best - position) * delta

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        pop_values = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(pop_values)
        best = pop[best_idx]
        best_value = pop_values[best_idx]

        eval_count = self.pop_size

        while eval_count < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, pop[i])
                
                if np.random.rand() < self.adaptive_threshold:
                    trial = self.quantum_tunneling(trial, best)
                
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                trial_value = func(trial)
                eval_count += 1

                if trial_value < pop_values[i]:
                    new_pop.append(trial)
                    pop_values[i] = trial_value
                    if trial_value < best_value:
                        best = trial
                        best_value = trial_value
                else:
                    new_pop.append(pop[i])

                if eval_count >= self.budget:
                    break
            
            pop = np.array(new_pop)
            
            # Adapt population size dynamically
            if eval_count < self.budget * 0.5:
                self.pop_size = int(self.base_pop_size * (1 + 0.2 * (eval_count / self.budget)))
            elif eval_count < self.budget * 0.8:
                self.pop_size = int(self.base_pop_size * (1 - 0.2 * (eval_count / self.budget)))

        return best