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
            # Evaluate current population
            scores = np.array([func(ind) for ind in self.pop])
            evaluations += self.population_size
            
            # Update the best solution
            if np.max(scores) > self.best_score:
                self.best_score = np.max(scores)
                self.best = self.pop[np.argmax(scores)]

            # Calculate diversity measure
            diversity = np.std(self.pop, axis=0).mean()
            
            # Dynamic F and CR based on diversity
            F = 0.5 if diversity > 0.1 else 0.9
            CR = 0.9 if diversity > 0.1 else 0.5
            
            # Adjust population size based on diversity
            self.population_size = max(int(self.population_size * (1 + 0.1 * (0.1 - diversity))), 1)
            self.pop = np.resize(self.pop, (self.population_size, self.dim))  # Maintain array size
            
            # Quantum-inspired mutation: Generate new population
            new_pop = np.zeros_like(self.pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[indices]
                mutation_vector = x0 + F * (x1 - x2)
                quantum_step = np.random.normal(0, 1, self.dim)
                trial_vector = np.clip(mutation_vector + quantum_step, 0, 1)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                new_pop[i] = np.where(crossover_mask, trial_vector, self.pop[i])
            
            # Evaluate new population
            new_scores = np.array([func(ind) for ind in new_pop])
            evaluations += self.population_size
            
            # Selection process
            for i in range(self.population_size):
                if new_scores[i] > scores[i]:
                    self.pop[i] = new_pop[i]
        
        return self.best