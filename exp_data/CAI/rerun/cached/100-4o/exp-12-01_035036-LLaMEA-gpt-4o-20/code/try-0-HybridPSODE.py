import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 20
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.5   # inertia weight
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        # Initialize PSO
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest_positions = np.copy(pop)
        pbest_scores = np.full(self.pop_size, np.inf)
        gbest_position = None
        gbest_score = np.inf
        
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.pop_size):
                score = func(pop[i])
                evaluations += 1
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = pop[i]
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = pop[i]

                if evaluations >= self.budget:
                    break

            # PSO update
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_positions - pop) +
                          self.c2 * r2 * (gbest_position - pop))
            pop = pop + velocities
            pop = np.clip(pop, self.bounds[0], self.bounds[1])
            
            # DE mutation and crossover
            for i in range(self.pop_size):
                indices = np.random.choice(range(self.pop_size), 3, replace=False)
                a, b, c = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                donor_vector = a + self.mutation_factor * (b - c)
                trial_vector = np.copy(pop[i])
                
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial_vector[j] = donor_vector[j]

                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])
                trial_score = func(trial_vector)
                evaluations += 1
                
                if trial_score < pbest_scores[i]:
                    pop[i] = trial_vector
                    pbest_scores[i] = trial_score
                    pbest_positions[i] = trial_vector
                    if trial_score < gbest_score:
                        gbest_score = trial_score
                        gbest_position = trial_vector

                if evaluations >= self.budget:
                    break

        return gbest_position, gbest_score