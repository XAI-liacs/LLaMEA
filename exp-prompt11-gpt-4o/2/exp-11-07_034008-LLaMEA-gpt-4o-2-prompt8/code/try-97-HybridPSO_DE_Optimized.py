import numpy as np

class HybridPSO_DE_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = np.zeros(dim)
        self.gbest_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        F = 0.8
        CR = 0.9
        
        while self.evaluations < self.budget:
            # Pre-calculate random values for efficiency
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return self.gbest_position
                
                # Evaluate current particle
                score = func(self.particles[i])
                self.evaluations += 1

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i]

                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i]

                # Update velocities and positions using PSO
                self.velocities[i] = (w * self.velocities[i] + 
                                      c1 * r1[i] * (self.pbest_positions[i] - self.particles[i]) +
                                      c2 * r2[i] * (self.gbest_position - self.particles[i]))
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)
            
            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return self.gbest_position
                
                # Mutation
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[candidates[0]], self.particles[candidates[1]], self.particles[candidates[2]]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.particles[i])

                # Selection
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.pbest_scores[i]:
                    self.particles[i] = trial
                    self.pbest_scores[i] = trial_score
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial

        return self.gbest_position