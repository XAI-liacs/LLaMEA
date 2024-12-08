import numpy as np

class Enhanced_PSO_ADE_Optimizer_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 60
        self.mutation_factor = 0.85
        self.crossover_prob = 0.8
        self.inertia_weight = 0.55
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.4
        self.dynamic_adjustment_freq = 7  # Slightly more frequent adjustments
        self.local_search_prob = 0.1  # Introduced local search probability

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        eval_count = 0
        iteration = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            eval_count += self.swarm_size

            for i in range(self.swarm_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]

            if iteration % self.dynamic_adjustment_freq == 0:
                self.inertia_weight *= 0.92  # Slightly faster reduction
                self.mutation_factor = 0.87 if iteration < self.budget // 2 else 0.76  # Adaptive mutation factor

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.swarm_size):
                if np.random.rand() < self.local_search_prob:  # Local search phase
                    indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 2, replace=False)
                    x0, x1 = positions[indices]
                    local_search_vector = np.clip(x0 + 0.5 * (x1 - positions[i]), self.lower_bound, self.upper_bound)
                    trial_score = func(local_search_vector)
                else:
                    indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                    x0, x1, x2 = positions[indices]
                    mutated_vector = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lower_bound, self.upper_bound)
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutated_vector, positions[i])
                    trial_score = func(trial_vector)
                
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = local_search_vector if np.random.rand() < self.local_search_prob else trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = local_search_vector if np.random.rand() < self.local_search_prob else trial_vector
                    global_best_score = trial_score

                if eval_count >= self.budget:
                    break
            
            iteration += 1

        return global_best_position, global_best_score