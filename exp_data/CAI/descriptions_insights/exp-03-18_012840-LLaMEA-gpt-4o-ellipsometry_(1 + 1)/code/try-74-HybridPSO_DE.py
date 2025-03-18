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
        stagnation_counter = 0  # New variable to track stagnation
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
                    stagnation_counter = 0  # Reset stagnation counter when improvement is found

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = current_position

            # PSO Update with adaptive inertia
            w = 0.5 + (0.9 - 0.5) * np.cos(evals / (0.3 * self.budget) * np.pi)  # Adjusted line
            c1 = 1.5 + np.random.rand() * 0.5
            c2 = 1.5
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)

            self.velocities = w * self.velocities + \
                              c1 * r1 * (self.personal_best_positions - self.particles) + \
                              c2 * r2 * (self.global_best_position - self.particles)
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, lb, ub)

            # DE Mutation and Crossover with dynamic mutation strategy
            F_base = 0.5 + 0.5 * np.cos(evals / self.budget * np.pi)  # Adjusted line
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                F = F_base if stagnation_counter < 5 else 0.9  # Adaptive F based on stagnation
                mutant_vector = np.clip(a + F * (b - c), lb, ub)

                crossover_rate = 0.7 + 0.3 * np.sin(evals / self.budget * np.pi)  # Adjusted line
                random_vector = np.random.rand(self.dim)
                trial_vector = np.where(random_vector < crossover_rate, mutant_vector, self.particles[i])
                
                trial_score = func(trial_vector)
                evals += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                    stagnation_counter = 0  # Reset stagnation counter when improvement is found

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

                stagnation_counter += 1  # Increment stagnation counter if no improvement

        return self.global_best_position, self.global_best_score