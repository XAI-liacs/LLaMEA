import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # PSO cognitive constant
        self.c2 = 1.5  # PSO social constant
        self.w = 0.7   # PSO inertia weight
        self.F = 0.5   # DE differential weight
        self.CR = 0.9  # DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, 
                                       self.upper_bound, 
                                       (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_idx])
        eval_count = self.population_size

        while eval_count < self.budget:
            # Differential Evolution Step
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(population[i])
                
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]
                
                trial_score = func(trial)
                eval_count += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                
                if trial_score < func(global_best_position):
                    global_best_position = trial

            # Particle Swarm Optimization Step
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                score = func(population[i])
                eval_count += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                
                if score < func(global_best_position):
                    global_best_position = population[i]

            if eval_count >= self.budget:
                break
        
        return global_best_position