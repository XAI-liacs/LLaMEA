import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.c1 = 1.4  # Cognitive component (changed)
        self.c2 = 1.5  # Social component
        self.inertia = 0.8  # Inertia weight
        self.f = 0.6  # Differential Evolution scaling factor (changed)
        self.cr = 0.75  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.pop_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf

        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            score = func(positions[i])
            self.evaluations += 1
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                # Adjust social component based on population diversity
                diversity_factor = np.std(positions, axis=0).mean() / (ub - lb).mean()
                adapted_c2 = self.c2 * (1 + 0.5 * diversity_factor)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 adapted_c2 * (1 + 0.7 * np.sin(np.pi * self.evaluations / self.budget) * np.random.randn()) * r2 * (global_best_position - positions[i]) +
                                 0.1 * np.random.randn(self.dim)) * (0.95 ** (self.evaluations / self.budget))
                velocities[i] = np.clip(velocities[i], -0.5 * (ub - lb), 0.5 * (ub - lb))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = positions[indices]
                self.f = 0.6 * (1 - self.evaluations / self.budget)
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr * (0.5 + 0.5 * np.cos(np.pi * self.evaluations / self.budget))
                trial = np.where(crossover, mutant, positions[i])

                if self.evaluations < self.budget:
                    score = func(trial)
                    self.evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = trial
                    
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = trial

            self.inertia = max(0.1, 0.8 - 0.7 * np.sqrt(self.evaluations / self.budget))

            self.pop_size = max(5, int(10 * self.dim * (1 - self.evaluations / self.budget)))

        return global_best_position