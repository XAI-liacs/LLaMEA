import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.pop_size, float('inf'))
        self.gbest_position = np.zeros(self.dim)
        self.gbest_score = float('inf')
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Evaluate current population
                if self.eval_count < self.budget:
                    fitness = func(self.population[i])
                    self.eval_count += 1
                    if fitness < self.pbest_scores[i]:
                        self.pbest_scores[i] = fitness
                        self.pbest_positions[i] = self.population[i]
                        if fitness < self.gbest_score:
                            self.gbest_score = fitness
                            self.gbest_position = self.population[i]

            # Differential Evolution strategy
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(self.population[a] + self.mutation_factor * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, self.population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = trial_fitness
                    self.pbest_positions[i] = trial
                    if trial_fitness < self.gbest_score:
                        self.gbest_score = trial_fitness
                        self.gbest_position = trial

            # Particle Swarm Optimization strategy
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.gbest_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)
        
        return self.gbest_position, self.gbest_score