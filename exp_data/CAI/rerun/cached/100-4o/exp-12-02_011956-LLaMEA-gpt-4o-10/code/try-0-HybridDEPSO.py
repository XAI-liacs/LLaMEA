import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.F = 0.8  # Differential evolution mutation factor
        self.CR = 0.9  # Crossover probability
        self.w = 0.5   # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive (personal) coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.max_evals = budget
        self.evals = 0
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')

    def _update_global_best(self):
        for i, score in enumerate(self.personal_best_scores):
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.personal_best_positions[i]

    def _differential_evolution(self, func):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
            trial_vector = np.copy(self.population[i])
            crossover = np.random.rand(self.dim) < self.CR
            trial_vector[crossover] = mutant_vector[crossover]
            trial_score = func(trial_vector)
            self.evals += 1
            if trial_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = trial_score
                self.personal_best_positions[i] = trial_vector

    def _particle_swarm_optimization(self, func):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.population)
        social = self.c2 * r2 * (self.global_best_position - self.population)
        self.velocities = self.w * self.velocities + cognitive + social
        self.population += self.velocities
        self.population = np.clip(self.population, self.bounds[0], self.bounds[1])

        for i in range(self.population_size):
            score = func(self.population[i])
            self.evals += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]

    def __call__(self, func):
        while self.evals < self.max_evals:
            self._differential_evolution(func)
            self._update_global_best()
            if self.evals < self.max_evals:  # Check to ensure budget is not exceeded
                self._particle_swarm_optimization(func)
                self._update_global_best()
        return self.global_best_position, self.global_best_score