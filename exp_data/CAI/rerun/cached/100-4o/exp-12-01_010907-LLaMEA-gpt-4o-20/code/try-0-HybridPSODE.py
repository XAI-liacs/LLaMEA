import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.randint(self.pop_size)]
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_score = np.inf
        self.eval_count = 0
    
    def __call__(self, func):
        def evaluate(particle):
            score = func(particle)
            return score
        
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Evaluate current particle
                score = evaluate(self.particles[i])
                self.eval_count += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update velocities and positions using PSO
            inertia_weight = 0.5
            cognitive_comp = 1.5
            social_comp = 1.5
            r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
            for i in range(self.pop_size):
                self.velocities[i] = (inertia_weight * self.velocities[i] + 
                                      cognitive_comp * r1 * (self.personal_best[i] - self.particles[i]) +
                                      social_comp * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

            # Perform DE mutation and crossover
            F = 0.8
            CR = 0.9
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, self.particles[i])
                
                # Evaluate trial vector
                trial_score = evaluate(trial)
                self.eval_count += 1

                # Selection
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.particles[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial

        return self.global_best