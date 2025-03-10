import numpy as np

class PSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.v_max = 0.1
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.uniform(-self.v_max, self.v_max, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F = 0.8  # DE mutation factor (changed from 0.7)
        self.CR = 0.7  # DE crossover rate

    def __call__(self, func):
        bounds = func.bounds
        eval_count = 0
        
        while eval_count < self.budget:
            # Evaluate particles
            for i in range(self.population_size):
                score = func(self.particles[i])
                eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                if eval_count >= self.budget:
                    break
            
            # Update velocities and positions (PSO)
            w = 0.9 - 0.3 * (eval_count / self.budget)  # Adjusted dynamic inertia weight
            c1, c2 = 1.5, 1.5  # cognitive and social coefficients
            velocity_mean = np.mean(self.velocities, axis=0)  # Historical velocity average
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social = c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social + 0.05 * velocity_mean  # Modified line
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)
            
            # Differential Evolution mutation and crossover
            self.CR = 0.7 + 0.3 * (eval_count / self.budget)  # Dynamic crossover rate adjustment
            self.F = 0.7 + 0.4 * (1 - eval_count / self.budget)  # Adaptive mutation factor
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                perturbation = np.random.uniform(-0.05, 0.05, self.dim)  # New line for random perturbation
                mutant = self.personal_best_positions[a] + self.F * (self.personal_best_positions[b] - self.personal_best_positions[c]) + 0.1 * (self.particles[a] - self.particles[b]) + perturbation
                mutant = np.clip(mutant, bounds.lb, bounds.ub)
                trial = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]
                trial_score = func(trial)
                eval_count += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial
                if eval_count >= self.budget:
                    break
        
        return self.global_best_position, self.global_best_score