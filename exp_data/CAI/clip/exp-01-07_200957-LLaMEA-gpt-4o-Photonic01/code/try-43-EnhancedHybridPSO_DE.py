import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased initial population size
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.8  # Adjusted inertia weight
        self.cognitive_weight = 1.8  # Adjusted cognitive weight
        self.social_weight = 2.1  # Adjusted social weight
        self.mutation_factor = 0.65  # Increased mutation factor
        self.crossover_rate = 0.85  # Adjusted crossover rate
        self.stagnation_count = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        generation = 0
        
        while eval_count < self.budget:
            if generation % 5 == 0:
                self.population_size = max(10, int(25 - (15 * eval_count / self.budget)))  # Adjusted resizing strategy
                self.particles = self.particles[:self.population_size]
                self.velocities = self.velocities[:self.population_size]
                self.personal_best_positions = self.personal_best_positions[:self.population_size]
                self.personal_best_scores = self.personal_best_scores[:self.population_size]

            prev_best_score = self.global_best_score

            for i in range(self.population_size):
                score = func(np.clip(self.particles[i], lb, ub))
                eval_count += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
            
            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.cognitive_weight * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.social_weight * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component) * np.random.uniform(0.8, 1.2)  # Added variability
                self.particles[i] += self.velocities[i]
                
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c) + self.mutation_factor * (np.random.rand(self.dim) - 0.5), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.particles[i])
                    trial_score = func(trial)
                    eval_count += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial
            
            self.inertia_weight = 0.5 + 0.5 * (1 - eval_count / self.budget)  # Adjusted inertia dynamic
            self.cognitive_weight = 1.5 + 1.5 * (1 - eval_count / self.budget) * (0.5 + 0.5 * np.cos(np.pi * generation / self.budget))  # Adjusted cognitive dynamic
            self.social_weight = 1.5 + 1.5 * (eval_count / self.budget)  # Adjusted social dynamic
            self.mutation_factor = (0.5 + 0.15 * np.sin(np.pi * generation / 9)) * (1 + 0.2 * self.stagnation_count)  # Fine-tuned mutation dynamic
            
            if self.global_best_score == prev_best_score:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
            
            if self.stagnation_count > 7:  # Adjusted stagnation threshold
                turbulence = np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))
                self.particles += turbulence * (ub - lb)
                self.stagnation_count = 0
            
            generation += 1
            
            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_score