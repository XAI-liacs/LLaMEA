import numpy as np

class AdaptiveDifferentialMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particles = np.random.rand(self.population_size, self.dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        generation = 0
        
        while eval_count < self.budget:
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
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.particles[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial
            
            # Adaptive mutation factor
            self.mutation_factor = 0.5 + 0.3 * (1 - eval_count / self.budget) * np.random.rand()
            
            # Preserve spatial diversity
            diversity = np.mean(np.std(self.particles, axis=0))
            if diversity < self.diversity_threshold:
                turbulence = np.random.uniform(-0.2, 0.2, size=(self.population_size, self.dim))
                self.particles += turbulence * (ub - lb)
            
            generation += 1
            
            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_score