import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4    # final inertia weight
        self.f = 0.8   # differential weight
        self.cr = 0.9  # crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        
        while evaluations < self.budget:
            self.w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / (self.budget * 0.8))  # Change 1

            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Particle Swarm Optimization (PSO) update
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i] if global_best_position is not None else 0))
                # Enhanced velocity perturbation
                if np.random.rand() < 0.3:
                    velocities[i] += np.random.normal(0, 0.05, self.dim) 
                
                candidate_position = np.clip(population[i] + velocities[i], lb, ub)
                candidate_score = func(candidate_position)
                evaluations += 1

                # Update personal best
                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate_position

                # Update global best
                if candidate_score < global_best_score:
                    global_best_score = candidate_score
                    global_best_position = candidate_position

            # Differential Evolution (DE) mutation and crossover
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Introduced stochastic adaptive change
                self.f = np.random.uniform(0.5, 1.0) + np.random.normal(0, 0.1)

                mutant_vector = np.clip(personal_best_positions[a] + self.f * (personal_best_positions[b] - personal_best_positions[c]), lb, ub)
                
                trial_vector = np.copy(personal_best_positions[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == jrand:
                        trial_vector[j] = mutant_vector[j]
                
                trial_score = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

            # Dynamic population size adjustment
            if evaluations % (self.budget // 10) == 0:
                population_size = max(10, population_size - 5 + int(np.random.rand() > 0.7))  # Change 2

        return global_best_position, global_best_score