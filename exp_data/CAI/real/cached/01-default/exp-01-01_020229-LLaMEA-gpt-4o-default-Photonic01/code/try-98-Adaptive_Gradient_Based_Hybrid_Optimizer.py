import numpy as np

class Adaptive_Gradient_Based_Hybrid_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.01  # Learning rate for gradient adjustment
        self.beta = 0.9  # Momentum factor for gradient-based search
        self.crossover_rate = 0.5
        self.mutation_scale = 0.1
        self.gradient_steps = int(budget * 0.2)  # Fraction of budget for gradient search
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Gradient-based local search
            for _ in range(self.gradient_steps):
                for i in range(self.population_size):
                    gradient = self.compute_gradient(func, position[i], lb, ub)
                    velocities[i] = self.beta * velocities[i] - self.alpha * gradient
                    position[i] += velocities[i]
                    position[i] = np.clip(position[i], lb, ub)

                    current_value = func(position[i])
                    evaluations += 1

                    if current_value < personal_best_value[i]:
                        personal_best_position[i] = position[i]
                        personal_best_value[i] = current_value

                    if current_value < global_best_value:
                        global_best_position = position[i]
                        global_best_value = current_value

                    if evaluations >= self.budget:
                        break
            
            # Evolutionary operations
            for i in range(0, self.population_size, 2):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < self.crossover_rate:
                    parent1 = personal_best_position[i]
                    parent2 = personal_best_position[(i + 1) % self.population_size]
                    offspring1, offspring2 = self.crossover(parent1, parent2)

                    self.evaluate_and_update(offspring1, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub)
                    self.evaluate_and_update(offspring2, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub)

            # Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                mutation = self.mutation_scale * np.random.normal(size=self.dim)
                mutated_position = personal_best_position[i] + mutation
                mutated_position = np.clip(mutated_position, lb, ub)

                self.evaluate_and_update(mutated_position, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub)

        return global_best_position, global_best_value

    def compute_gradient(self, func, position, lb, ub, epsilon=1e-6):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            pos_eps = np.copy(position)
            neg_eps = np.copy(position)
            pos_eps[i] += epsilon
            neg_eps[i] -= epsilon
            pos_eps = np.clip(pos_eps, lb, ub)
            neg_eps = np.clip(neg_eps, lb, ub)
            gradient[i] = (func(pos_eps) - func(neg_eps)) / (2 * epsilon)
        return gradient

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2

    def evaluate_and_update(self, position, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub):
        current_value = func(position)
        evaluations += 1

        idx = np.argmin(personal_best_value)
        if current_value < personal_best_value[idx]:
            personal_best_position[idx] = position
            personal_best_value[idx] = current_value

        if current_value < global_best_value:
            global_best_position = position
            global_best_value = current_value