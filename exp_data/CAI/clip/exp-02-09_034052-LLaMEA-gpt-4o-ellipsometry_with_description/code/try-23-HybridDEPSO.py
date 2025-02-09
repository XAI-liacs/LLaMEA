import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.omega = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component for PSO
        self.c2 = 1.5  # Social component for PSO
        self.population = []
        self.velocities = []

    def initialize_population(self, lb, ub):
        for _ in range(self.population_size):
            individual = np.random.uniform(lb, ub, self.dim)
            score = float('inf')
            velocity = np.random.uniform(-abs(ub-lb), abs(ub-lb), self.dim)
            self.population.append((individual, score))
            self.velocities.append(velocity)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        best_solution = None
        best_score = float('inf')
        global_best = np.random.uniform(lb, ub, self.dim)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution step
                target_vector, target_score = self.population[i]
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                x1, _ = self.population[a]
                x2, _ = self.population[b]
                x3, _ = self.population[c]

                mutant_vector = x1 + self.f * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                trial_vector = np.empty(self.dim)
                j_rand = np.random.randint(self.dim)

                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = target_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < target_score:
                    self.population[i] = (trial_vector, trial_score)

                    if trial_score < best_score:
                        best_solution = trial_vector
                        best_score = trial_score

                # Particle Swarm Optimization step
                new_velocity = (self.omega * self.velocities[i] + 
                                self.c1 * np.random.rand(self.dim) * (target_vector - trial_vector) + 
                                self.c2 * np.random.rand(self.dim) * (global_best - trial_vector))
                
                new_position = trial_vector + new_velocity
                new_position = np.clip(new_position, lb, ub)
                new_score = func(new_position)
                evaluations += 1

                if new_score < trial_score:
                    self.population[i] = (new_position, new_score)
                    self.velocities[i] = new_velocity

                    if new_score < best_score:
                        best_solution = new_position
                        best_score = new_score
                        global_best = new_position

                # Adaptive parameter control
                if evaluations % (self.budget // 10) == 0:
                    self.f = np.random.uniform(0.4, 0.9)
                    self.cr = np.random.uniform(0.5, 1.0)
                    self.omega = np.random.uniform(0.4, 0.9)

        return best_solution