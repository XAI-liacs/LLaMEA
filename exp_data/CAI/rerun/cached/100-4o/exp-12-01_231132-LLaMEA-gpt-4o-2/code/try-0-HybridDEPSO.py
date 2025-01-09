import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.7   # inertia weight
        self.F = 0.5   # differential weight
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evals = self.population_size
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                donor_vector = (population[indices[0]]
                                + self.F * (population[indices[1]] - population[indices[2]]))
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, donor_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                # Particle Swarm Update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - population[i])
                                 + self.c2 * r2 * (global_best_position - population[i]))
                new_position = population[i] + velocities[i]
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)
                new_score = func(new_position)
                evals += 2

                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = new_position
                    population[i] = new_position
                
                if personal_best_scores[i] < global_best_score:
                    global_best_score = personal_best_scores[i]
                    global_best_position = personal_best_positions[i]

                if evals >= self.budget:
                    break
                    
        return global_best_position, global_best_score