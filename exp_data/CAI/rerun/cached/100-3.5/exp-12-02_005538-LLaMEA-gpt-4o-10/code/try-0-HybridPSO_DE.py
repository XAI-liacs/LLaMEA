import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.f_scale = 0.5
        self.cr = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.population[i])
                evaluations += 1
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.population[i]
                    
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.population[i]
                    
                if evaluations >= self.budget:
                    break
            
            # PSO Update
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (self.inertia_weight * self.velocities +
                               self.cognitive_const * r1 * (self.pbest_positions - self.population) +
                               self.social_const * r2 * (self.gbest_position - self.population))
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)
            
            # DE Update
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f_scale * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < self.pbest_scores[i]:
                    self.pbest_scores[i] = trial_score
                    self.pbest_positions[i] = trial
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial

        return self.gbest_position