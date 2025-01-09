import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.iteration = 0

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                score = func(self.population[i])
                evaluations += 1
                
                # Update personal best
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.population[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

            # Switch between DE and PSO strategies
            if evaluations < self.budget * 0.5:
                self._differential_evolution_step(func)
            else:
                self._particle_swarm_optimization_step()
            self.iteration += 1

        return self.global_best_position

    def _differential_evolution_step(self, func):
        F = 0.8
        CR = 0.9
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), 
                             self.lower_bound, self.upper_bound)
            trial = np.copy(self.population[i])
            for j in range(self.dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]
            trial_score = func(trial)
            if trial_score < self.best_scores[i]:
                self.population[i] = trial
                self.best_scores[i] = trial_score

    def _particle_swarm_optimization_step(self):
        w = 0.5
        c1 = 1.5
        c2 = 1.5
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocity[i] = (w * self.velocity[i] +
                                c1 * r1 * (self.best_positions[i] - self.population[i]) +
                                c2 * r2 * (self.global_best_position - self.population[i]))
            self.population[i] = np.clip(self.population[i] + self.velocity[i],
                                         self.lower_bound, self.upper_bound)