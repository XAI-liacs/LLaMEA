import numpy as np

class EnhancedQuantumPSODE:
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
        self.inertia = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.secondary_swarm = np.random.rand(self.particle_count, dim)
        self.secondary_scores = np.full(self.particle_count, np.inf)
        self.secondary_gbest = None
        self.secondary_gbest_score = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        func_eval_count = 0
        perturbation_scale = 0.1
        velocity_bound = 0.1 * (ub - lb)
        convergence_threshold = 1e-4  # Added line: Introduce threshold for convergence
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

            for i in range(self.particle_count):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.inertia * self.velocities[i] +
                                     (1.2 + 0.3 * np.sin(3.14 * func_eval_count / self.budget)) * r1 * (self.pbest[i] - self.pop[i]) +
                                     self.social_coeff * r2 * (self.gbest - self.pop[i]))
                self.velocities[i] = np.clip(self.velocities[i], -velocity_bound, velocity_bound)
                self.pop[i] += self.velocities[i]

                if np.random.rand() < 0.9:  # Modified Crossover Probability
                    a, b, c = np.random.choice(self.particle_count, 3, replace=False)
                    mutant = self.pbest[a] + 0.8 * (self.pbest[b] - self.pbest[c])  # Modified Mutation Factor
                    trial = np.where(np.random.rand(self.dim) < 0.9, mutant, self.pop[i])  # Modified Crossover Rate
                    trial = np.clip(trial, lb, ub)
                    trial_score = func(trial)
                    func_eval_count += 1
                    if trial_score < score:
                        self.pop[i] = trial

            self.secondary_swarm += np.random.normal(0, perturbation_scale, self.secondary_swarm.shape)
            perturbation_scale *= 0.98  # Modified Perturbation Decay
            for i in range(self.particle_count):
                self.secondary_swarm[i] = np.clip(self.secondary_swarm[i], lb, ub)
                secondary_score = func(self.secondary_swarm[i])
                func_eval_count += 1
                if secondary_score < self.secondary_scores[i]:
                    self.secondary_scores[i] = secondary_score
                    if secondary_score < self.secondary_gbest_score:
                        self.secondary_gbest_score = secondary_score
                        self.secondary_gbest = self.secondary_swarm[i].copy()

            self.inertia = 0.7 + 0.3 * np.sin(3.14 * func_eval_count / self.budget)
            # Adjust exploration when convergence is detected
            if func_eval_count / self.budget > 0.5 and np.mean(self.pbest_scores) - self.gbest_score < convergence_threshold:
                self.pop += np.random.normal(0, 0.1, self.pop.shape)  # Modified Exploration Strategy

            if self.gbest_score > self.secondary_gbest_score:
                self.gbest_score = self.secondary_gbest_score
                self.gbest = self.secondary_gbest.copy()

        return self.gbest