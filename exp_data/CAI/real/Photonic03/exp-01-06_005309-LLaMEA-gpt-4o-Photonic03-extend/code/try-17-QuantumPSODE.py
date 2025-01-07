import numpy as np

class QuantumPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 12 + 2 * int(np.sqrt(dim))  # Changed initialization strategy
        self.pop = np.random.rand(self.particle_count, dim)
        self.velocities = np.zeros_like(self.pop)
        self.pbest = np.copy(self.pop)
        self.pbest_scores = np.full(self.particle_count, np.inf)
        self.gbest = None
        self.gbest_score = np.inf
        self.inertia = 0.8  # Adjusted inertia for improved balance
        self.cognitive_coeff = 2.0  # Enhanced learning component
        self.social_coeff = 1.3
        self.de_mutation_factor = 0.8  # Adaptive mutation factor
        self.de_crossover_rate = 0.85  # Modified crossover rate for exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        func_eval_count = 0
        momentum = np.zeros(self.dim)  # Added momentum component
        alpha = 0.01  # Momentum learning rate
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

            for i in range(self.particle_count):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.inertia * self.velocities[i] +
                                     self.cognitive_coeff * r1 * (self.pbest[i] - self.pop[i]) +
                                     self.social_coeff * r2 * (self.gbest - self.pop[i]))
                self.pop[i] += self.velocities[i] + alpha * momentum  # Integrated momentum
                momentum = 0.9 * momentum + self.velocities[i]

                if np.random.rand() < self.de_crossover_rate:
                    a, b, c = np.random.choice(self.particle_count, 3, replace=False)
                    mutant = self.pbest[a] + self.de_mutation_factor * (self.pbest[b] - self.pbest[c])
                    trial = np.where(np.random.rand(self.dim) < self.de_crossover_rate, mutant, self.pop[i])
                    trial = np.clip(trial, lb, ub)
                    trial_score = func(trial)
                    func_eval_count += 1
                    if trial_score < score:
                        self.pop[i] = trial
            
            self.inertia = 0.6 + 0.4 * np.sin(3.14 * func_eval_count / self.budget)  # Adjusted inertia schedule
            if func_eval_count / self.budget > 0.5 and np.mean(self.pbest_scores) - self.gbest_score < 1e-5:
                self.pop += np.random.normal(0, 0.1, self.pop.shape)  # Adjust noise intensity for escaping local minima

        return self.gbest