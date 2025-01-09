import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.global_best_position = self.population[0]
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf
        self.f_evals = 0

    def __call__(self, func):
        while self.f_evals < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                score = func(self.population[i])
                self.f_evals += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

            # Update velocities and positions using PSO
            inertia_weight = 0.5 + np.random.rand() / 2.0
            personal_acceleration = 1.5 * np.random.rand()
            global_acceleration = 1.5 * np.random.rand()
            for i in range(self.population_size):
                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    personal_acceleration * np.random.rand() * 
                    (self.personal_best_positions[i] - self.population[i]) +
                    global_acceleration * np.random.rand() * 
                    (self.global_best_position - self.population[i])
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)

            # Apply mutation and crossover from DE
            for i in range(self.population_size):
                if np.random.rand() < 0.8:  # Reduced crossover probability for better balance
                    candidates = list(range(self.population_size))
                    candidates.remove(i)
                    a, b, c = np.random.choice(candidates, 3, replace=False)
                    mutant = self.population[a] + 0.8 * (self.population[b] - self.population[c])
                    mutant = np.clip(mutant, -5.0, 5.0)
                    trial = np.copy(self.population[i])
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < 0.9 or j == j_rand:
                            trial[j] = mutant[j]
                    if func(trial) < func(self.population[i]):
                        self.population[i] = trial

        return self.global_best_position, self.global_best_score