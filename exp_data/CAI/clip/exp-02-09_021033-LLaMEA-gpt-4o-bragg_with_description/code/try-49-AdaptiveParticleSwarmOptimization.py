import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.initial_swarm_size = 10 * dim
        self.inertia_weight = 0.9  # Original: 0.9
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.adaptive_local_search_rate = 0.4  # Increased local search rate
        self.exploration_exploitation_balance = 0.5
        self.bounds = None
        
    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        swarm_positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm_velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = swarm_positions.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_positions)
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]
        
        evaluations = self.swarm_size

        while evaluations < self.budget:
            self.swarm_size = int(self.initial_swarm_size * (1 - evaluations / self.budget) * (global_best_fitness / np.max(personal_best_fitness))) + 1  # Dynamic reduction
            self.exploration_exploitation_balance = 0.5 + 0.5 * np.sin(np.pi * evaluations / self.budget)  # Nonlinear balance
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update inertia weight dynamically with stochastic decay
                self.inertia_weight = 0.4 + 0.3 * np.exp(-evaluations / (self.budget + np.random.rand())) + 0.1 * np.sin(2 * np.pi * evaluations / self.budget) # Changed to exponential decay + oscillation

                # Update cognitive coefficient dynamically
                self.c1 = 1.5 + 0.5 * np.cos(np.pi * evaluations / self.budget)  # Nonlinear cognitive adjustment

                # Update social coefficient dynamically
                self.c2 = 1.5 + 0.5 * np.sin(np.pi * evaluations / self.budget)  # Nonlinear social adjustment

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - swarm_positions[i])
                social_component = self.c2 * r2 * (global_best_position - swarm_positions[i])
                swarm_velocities[i] = (self.inertia_weight * swarm_velocities[i] + cognitive_component + 
                                       social_component * self.exploration_exploitation_balance + np.random.normal(0, 0.1, self.dim))  # Added stochastic element
                swarm_velocities[i] *= 0.95 * (1 - 0.5 * np.sin(np.pi * evaluations / self.budget))  # Adaptive damping factor

                # Update position
                swarm_positions[i] = np.clip(swarm_positions[i] + swarm_velocities[i], self.bounds[0], self.bounds[1])
                
                # Evaluate new position
                current_fitness = func(swarm_positions[i])
                evaluations += 1

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = swarm_positions[i]
                    personal_best_fitness[i] = current_fitness

                    # Update global best
                    if current_fitness < global_best_fitness:
                        global_best_position = swarm_positions[i]
                        global_best_fitness = current_fitness

                # Local search with stochastic perturbations
                if np.random.rand() < self.adaptive_local_search_rate:
                    local_trial = self.local_search(swarm_positions[i], func, evaluations)
                    evaluations += 1
                    local_trial_fitness = func(local_trial)
                    if local_trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = local_trial
                        personal_best_fitness[i] = local_trial_fitness
                        if local_trial_fitness < global_best_fitness:
                            global_best_position = local_trial
                            global_best_fitness = local_trial_fitness

        return global_best_position
    
    def local_search(self, position, func, evaluations):
        local = position.copy()
        adaptive_perturbation = 0.05 * np.exp(-evaluations / self.budget) # Reduced perturbation
        perturbation = np.random.normal(0, adaptive_perturbation, self.dim)
        local += perturbation
        local = np.clip(local, self.bounds[0], self.bounds[1])
        return local