import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.rand(self.pop_size, dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.bounds = None

    def _initialize(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        self.velocities = (ub - lb) * (np.random.rand(self.pop_size, self.dim) - 0.5) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self._initialize(lb, ub)
        
        evals = 0
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                current_position = self.particles[i]
                score = func(current_position)
                evals += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = current_position

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = current_position

            # PSO Update
            w = 0.4 + (0.9 - 0.4) * np.exp(-evals / (0.1 * self.budget)) * (self.global_best_score / np.mean(self.personal_best_scores))  # Adjusted line
            c1 = 1.5 + np.random.rand() * 0.5
            c2 = 1.5
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)

            self.velocities = w * (self.velocities + 
                                 c1 * r1 * (self.personal_best_positions - self.particles) +
                                 c2 * r2 * (self.global_best_position - self.particles)) * 0.9
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, lb, ub)
            
            # Random perturbation for diversity
            if evals < self.budget:
                perturbation_chance = 0.1 * (1 - evals / self.budget)
                perturbation = np.random.rand(self.pop_size, self.dim) < perturbation_chance
                self.particles += np.random.normal(0, 0.1, size=self.particles.shape) * perturbation

            # DE Mutation and Crossover
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + 0.3 * np.sin(evals / self.budget * np.pi)  # Adaptive F
                mutant_vector = np.clip(a + F * (b - c), lb, ub)

                crossover_rate = 0.8 + 0.2 * np.sin(evals / self.budget * np.pi)  # Adaptive crossover rate
                random_vector = np.random.rand(self.dim)
                trial_vector = np.where(random_vector < crossover_rate, mutant_vector, self.particles[i])
                
                trial_score = func(trial_vector)
                evals += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_score