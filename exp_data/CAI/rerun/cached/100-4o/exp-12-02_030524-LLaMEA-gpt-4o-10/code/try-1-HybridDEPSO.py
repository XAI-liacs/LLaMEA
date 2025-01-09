import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // (5 * dim))  # Adaptive population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7    # Inertia weight
        self.f = 0.5    # DE scaling factor
        self.cr = 0.9   # DE crossover probability
        
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            -(self.upper_bound - self.lower_bound), 
            self.upper_bound - self.lower_bound, 
            (self.population_size, self.dim)
        )
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.population_size
        while evaluations < self.budget:
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.cr
                trial[crossover] = mutant[crossover]
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score
                    
                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score
            
            # Particle Swarm Optimization update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best[i] - population[i])
                    + self.c2 * r2 * (global_best - population[i])
                )
                population[i] = np.clip(
                    population[i] + velocities[i], self.lower_bound, self.upper_bound
                )
                score = func(population[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = score
                    
                    if score < global_best_score:
                        global_best = population[i]
                        global_best_score = score
                
                if evaluations >= self.budget:
                    break
                
        return global_best, global_best_score