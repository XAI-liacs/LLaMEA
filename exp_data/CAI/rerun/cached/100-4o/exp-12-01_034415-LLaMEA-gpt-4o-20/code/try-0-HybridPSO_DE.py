import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 40
        self.inertia_weight = 0.5
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.de_cr = 0.9
        self.de_f = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_pos = np.copy(self.population)
        self.personal_best_val = np.full(self.pop_size, np.inf)
        self.global_best_pos = None
        self.global_best_val = np.inf
        self.eval_count = 0

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[i] = (self.inertia_weight * self.velocities[i] + 
                              self.cognitive_const * r1 * (self.personal_best_pos[i] - self.population[i]) +
                              self.social_const * r2 * (self.global_best_pos - self.population[i]))
        self.population[i] += self.velocities[i]
        self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
        value = func(self.population[i])
        self.eval_count += 1
        if value < self.personal_best_val[i]:
            self.personal_best_val[i] = value
            self.personal_best_pos[i] = self.population[i]
        if value < self.global_best_val:
            self.global_best_val = value
            self.global_best_pos = self.population[i]

    def differential_evolution(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            indices = list(range(self.pop_size))
            indices.remove(i)
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
            trial = np.where(np.random.rand(self.dim) < self.de_cr, mutant, self.population[i])
            value = func(trial)
            self.eval_count += 1
            if value < func(self.population[i]):
                self.population[i] = trial
                if value < self.personal_best_val[i]:
                    self.personal_best_val[i] = value
                    self.personal_best_pos[i] = trial
                if value < self.global_best_val:
                    self.global_best_val = value
                    self.global_best_pos = trial

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                self.update_particle(i, func)
            self.differential_evolution(func)
        return self.global_best_pos, self.global_best_val