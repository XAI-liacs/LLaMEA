import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = np.random.uniform(0, 1, (self.population_size, dim))
        self.best = None
        self.best_score = float('-inf')
        self.F = 0.5
        self.CR = 0.9

    def levy_flight(self, L):
        u = np.random.normal(0, 1, L)
        v = np.random.normal(0, 1, L)
        return u / np.power(abs(v), 1/3)  # Lévy distribution

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            scores = np.array([func(ind) for ind in self.pop])
            evaluations += self.population_size
            
            if np.max(scores) > self.best_score:
                self.best_score = np.max(scores)
                self.best = self.pop[np.argmax(scores)]

            diversity = np.std(self.pop, axis=0).mean()
            
            self.F = np.random.normal(0.7, 0.2) if diversity > 0.15 else np.random.normal(0.9, 0.05)  # Adjusted diversity threshold
            self.CR = np.random.normal(0.92, 0.08) if diversity > 0.15 else np.random.normal(0.65, 0.05)  # Refined CR adjustment
            self.F = np.clip(self.F * np.exp(-evaluations/self.budget) * (self.best_score / (self.best_score + 0.1)), 0, 1)  # Adjusted F based on score
            self.CR = np.clip(self.CR, 0, 1)
            
            new_pop = np.zeros_like(self.pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[indices]
                mutation_vector = x0 + self.F * (x1 - x2)
                quantum_step = self.levy_flight(self.dim)  # Incorporating Lévy flight
                trial_vector = np.clip(mutation_vector + quantum_step, 0, 1)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                new_pop[i] = np.where(crossover_mask, trial_vector, self.pop[i])
            
            new_scores = np.array([func(ind) for ind in new_pop])
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if new_scores[i] > scores[i]:
                    self.pop[i] = new_pop[i]
            
            improvement_rate = np.max(new_scores) / (self.best_score + 1e-9)
            if improvement_rate > 1.01:
                self.population_size = min(20 * self.dim, self.population_size * 2)
            elif improvement_rate < 0.99:
                self.population_size = max(5 * self.dim, self.population_size // 2)

        return self.best