import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 30
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.pop_size, float('inf'))
        self.gbest_position = np.zeros(self.dim)
        self.gbest_score = float('inf')
        self.f = 0.5 # DE mutation factor
        self.cr = 0.9 # DE crossover rate

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current particles
            for i in range(self.pop_size):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i].copy()
                
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i].copy()

                if evaluations >= self.budget:
                    break

            # Update velocities and positions using PSO mechanism
            inertia_weight = 0.7
            cognitive_component = 1.5
            social_component = 1.5
            
            r1, r2 = np.random.rand(2)
            self.velocities = (inertia_weight * self.velocities +
                               cognitive_component * r1 * (self.pbest_positions - self.particles) +
                               social_component * r2 * (self.gbest_position - self.particles))
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Differential Evolution operators
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(self.particles[a] + self.f * (self.particles[b] - self.particles[c]), 
                                 self.lower_bound, self.upper_bound)
                trial = np.copy(self.particles[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == jrand:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.pbest_scores[i]:
                    self.particles[i] = trial
                    self.pbest_scores[i] = trial_score
                    self.pbest_positions[i] = trial

                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial.copy()

                if evaluations >= self.budget:
                    break

        return self.gbest_position, self.gbest_score