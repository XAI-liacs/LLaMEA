import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = np.random.uniform(0, 1, (self.population_size, dim))
        self.best = None
        self.best_score = float('-inf')
        self.F_min, self.F_max = 0.3, 0.9
        self.CR = 0.9

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate current population
            scores = np.array([func(ind) for ind in self.pop])
            evaluations += self.population_size
            
            # Update the best solution
            if np.max(scores) > self.best_score:
                self.best_score = np.max(scores)
                self.best = self.pop[np.argmax(scores)]

            # Calculate diversity measure
            diversity = np.std(self.pop, axis=0).mean()
            
            # Dynamic F range adjustment
            self.F_min, self.F_max = (0.1, 0.6) if diversity > 0.2 else (0.4, 0.9)
            self.F = np.random.uniform(self.F_min, self.F_max)
            self.CR = np.random.uniform(0.8, 1.0) if diversity > 0.2 else np.random.uniform(0.5, 0.7)
            self.F = np.clip(self.F, 0, 1)
            self.CR = np.clip(self.CR, 0, 1)
            
            # Quantum-inspired mutation: Generate new population
            new_pop = np.zeros_like(self.pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[indices]
                mutation_vector = x0 + self.F * (x1 - x2)
                quantum_step = np.random.normal(0, 1, self.dim)
                trial_vector = np.clip(mutation_vector + quantum_step, 0, 1)
                
                # Crossover with adaptive probability
                crossover_mask = np.random.rand(self.dim) < self.CR
                new_pop[i] = np.where(crossover_mask, trial_vector, self.pop[i])
            
            # Evaluate new population
            new_scores = np.array([func(ind) for ind in new_pop])
            evaluations += self.population_size
            
            # Selection process
            for i in range(self.population_size):
                if new_scores[i] > scores[i]:
                    self.pop[i] = new_pop[i]
            
            # Adaptive population size based on progress
            if np.max(new_scores) > self.best_score * 0.95:
                self.population_size = max(5 * self.dim, self.population_size // 2)

        return self.best