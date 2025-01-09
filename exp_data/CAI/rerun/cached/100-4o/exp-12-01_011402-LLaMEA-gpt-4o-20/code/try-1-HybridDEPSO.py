import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + int(2 * np.sqrt(self.dim))
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9
    
    def __call__(self, func):
        np.random.seed()
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best_positions = population.copy()
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        
        while eval_count < self.budget:
            scores = np.array([func(ind) for ind in population])
            eval_count += self.population_size
            
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i].copy()
                    
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = population[i].copy()
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                trial_vector = np.where(
                    np.random.rand(self.dim) < self.CR,
                    a + self.F * (b - c),
                    population[i]
                )
                trial_vector = np.clip(trial_vector, self.lb, self.ub)
                trial_score = func(trial_vector)
                eval_count += 1
                
                if trial_score < scores[i]:
                    population[i] = trial_vector
                    scores[i] = trial_score
                    
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                               self.c2 * r2 * (global_best_position - population[i]))
                population[i] += velocity[i]
                population[i] = np.clip(population[i], self.lb, self.ub)
                
        return global_best_position, global_best_score