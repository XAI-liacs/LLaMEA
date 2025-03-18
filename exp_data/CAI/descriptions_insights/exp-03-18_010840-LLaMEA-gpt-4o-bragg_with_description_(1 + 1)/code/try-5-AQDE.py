import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = np.random.uniform(0, 1, (self.population_size, dim))
        self.best = None
        self.best_score = float('-inf')

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            scores = np.array([func(ind) for ind in self.pop])
            evaluations += self.population_size
            
            if np.max(scores) > self.best_score:
                self.best_score = np.max(scores)
                self.best = self.pop[np.argmax(scores)]

            diversity = np.std(self.pop, axis=0).mean()
            
            # Improved dynamic F and CR based on diversity
            F = 0.5 + 0.4 * (diversity < 0.1)
            CR = 0.9 - 0.4 * (diversity < 0.1)
            
            new_pop = np.zeros_like(self.pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[indices]
                mutation_vector = x0 + F * (x1 - x2)
                quantum_step = np.random.normal(0, 0.5, self.dim)  # Adjusted step size
                trial_vector = np.clip(mutation_vector + quantum_step, 0, 1)
                
                crossover_mask = np.random.rand(self.dim) < CR
                new_pop[i] = np.where(crossover_mask, trial_vector, self.pop[i])
            
            new_scores = np.array([func(ind) for ind in new_pop])
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if new_scores[i] > scores[i]:
                    self.pop[i] = new_pop[i]
        
        return self.best