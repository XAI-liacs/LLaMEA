import numpy as np

class QuantumHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.quantum_factor = 0.1
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.local_search_probability = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            # Update velocities and positions for PSO with quantum influence
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best - pop) + 
                          self.c2 * r2 * np.sin(global_best - pop) * self.quantum_factor)
            pop = pop + velocities
            pop = np.clip(pop, lb, ub)

            # Evaluate new population
            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best[i] = pop[i]
            
            if np.min(personal_best_scores) < global_best_score:
                global_best_score = np.min(personal_best_scores)
                global_best = personal_best[np.argmin(personal_best_scores)]

            # Quantum-inspired evolutionary process
            for idx in range(self.population_size):
                if np.random.rand() < self.local_search_probability:
                    a, b, c = pop[np.random.choice(np.arange(self.population_size), 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, pop[idx])

                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < scores[idx]:
                        scores[idx] = trial_score
                        pop[idx] = trial

        return global_best, global_best_score