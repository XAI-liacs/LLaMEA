import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50
        self.f = 0.5
        self.cr = 0.9
        self.c1 = 0.5
        self.c2 = 0.3
        self.w = 0.5
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_positions = np.copy(self.positions)
        self.global_best_position = np.copy(self.positions[np.argmin(self.evaluate_population())])
        self.eval_count = 0
    
    def evaluate_population(self):
        return np.array([func(ind) for ind in self.positions])
    
    def __call__(self, func):
        self.func = func
        while self.eval_count < self.budget:
            self.de_step()
            self.pso_step()
            self.update_global_best()
        return self.global_best_position

    def de_step(self):
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.positions[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, self.positions[i])
            if func(trial_vector) < func(self.positions[i]):
                self.positions[i] = trial_vector
                self.best_positions[i] = trial_vector
            self.eval_count += 1
            if self.eval_count >= self.budget:
                break

    def pso_step(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        self.velocities = self.w * self.velocities + self.c1 * r1 * (self.best_positions - self.positions) \
                          + self.c2 * r2 * (self.global_best_position - self.positions)
        self.positions = np.clip(self.positions + self.velocities, self.bounds[0], self.bounds[1])

    def update_global_best(self):
        current_best_idx = np.argmin(self.evaluate_population())
        if func(self.positions[current_best_idx]) < func(self.global_best_position):
            self.global_best_position = np.copy(self.positions[current_best_idx])