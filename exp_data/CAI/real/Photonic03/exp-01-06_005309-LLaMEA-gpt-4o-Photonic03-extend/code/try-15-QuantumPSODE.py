import numpy as np

class QuantumPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 10 + 3 * int(np.sqrt(dim))
        self.pop = np.random.rand(self.particle_count, dim)
        self.velocities = np.zeros_like(self.pop)
        self.pbest = np.copy(self.pop)
        self.pbest_scores = np.full(self.particle_count, np.inf)
        self.gbest = None
        self.gbest_score = np.inf
        self.inertia_initial = 0.9
        self.inertia_final = 0.4
        self.inertia = self.inertia_initial
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.de_mutation_factor = 0.9
        self.de_crossover_rate = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        func_eval_count = 0
        while func_eval_count < self.budget:
            for i in range(self.particle_count):
                self.pop[i] = np.clip(self.pop[i], lb, ub)
                score = func(self.pop[i])
                func_eval_count += 1
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.pop[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.pop[i].copy()

            if func_eval_count >= self.budget:
                break

            neighbors = np.random.choice(self.particle_count, 3, replace=False)
            for i in range(self.particle_count):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.inertia * self.velocities[i] +
                                     self.cognitive_coeff * r1 * (self.pbest[i] - self.pop[i]) +
                                     self.social_coeff * r2 * (self.gbest - self.pop[i]))
                self.pop[i] += self.velocities[i]

                if np.random.rand() < self.de_crossover_rate:
                    a, b, c = neighbors
                    mutant = self.pbest[a] + self.de_mutation_factor * (self.pbest[b] - self.pbest[c])
                    trial = np.where(np.random.rand(self.dim) < self.de_crossover_rate, mutant, self.pop[i])
                    trial = np.clip(trial, lb, ub)
                    trial_score = func(trial)
                    func_eval_count += 1
                    if trial_score < self.pbest_scores[i]:
                        self.pop[i] = trial

            self.inertia = (self.inertia_final + (self.inertia_initial - self.inertia_final) * 
                            ((self.budget - func_eval_count) / self.budget))
            if func_eval_count / self.budget > 0.5 and np.mean(self.pbest_scores) - self.gbest_score < 1e-5:
                self.pop += np.random.normal(0, 0.1 * (1 - func_eval_count / self.budget), self.pop.shape)

        return self.gbest