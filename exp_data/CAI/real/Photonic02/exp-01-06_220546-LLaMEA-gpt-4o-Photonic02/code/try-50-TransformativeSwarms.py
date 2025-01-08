import numpy as np

class TransformativeSwarms:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.num_swarms = 3  # Split population into multiple swarms
        self.inertia = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.local_search_prob = 0.3

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        velocity = np.zeros((self.population_size, self.dim))
        position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(ind) for ind in position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        
        eval_count = self.population_size
        iterations = 0

        while eval_count < self.budget:
            for swarm_index in range(self.num_swarms):
                # Select swarm-specific indices
                swarm_indices = np.array_split(np.arange(self.population_size), self.num_swarms)[swarm_index]
                r1, r2 = np.random.rand(2)
                # Dynamic learning coefficients
                self.cognitive_coef = 1.5 + 0.5 * np.random.rand()
                self.social_coef = 1.5 + 0.5 * np.random.rand()
                
                # Update velocity and position for the swarm
                velocity[swarm_indices] = (self.inertia * velocity[swarm_indices] +
                                           self.cognitive_coef * r1 * (personal_best_position[swarm_indices] - position[swarm_indices]) +
                                           self.social_coef * r2 * (global_best_position - position[swarm_indices]))
                position[swarm_indices] += velocity[swarm_indices]
                position[swarm_indices] = np.clip(position[swarm_indices], lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in position])
            eval_count += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if new_values[i] < personal_best_value[i]:
                    personal_best_value[i] = new_values[i]
                    personal_best_position[i] = position[i]
            global_best_position = personal_best_position[np.argmin(personal_best_value)]

            # Adaptive local search
            self.local_search_prob = 0.3 * (1 - (eval_count / self.budget))
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.normal(0, 0.1, self.dim)
                for i in range(self.population_size):
                    candidate = position[i] + perturbation
                    candidate = np.clip(candidate, lb, ub)
                    candidate_value = func(candidate)
                    eval_count += 1
                    if candidate_value < personal_best_value[i]:
                        personal_best_value[i] = candidate_value
                        personal_best_position[i] = candidate
            
            # Periodic swarm merging
            if iterations % 10 == 0:
                global_best_position = position[np.argmin([func(ind) for ind in position])]

            iterations += 1

        return global_best_position, func(global_best_position)