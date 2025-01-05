import numpy as np

class Swarm_Enhanced_Memetic_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.local_search_iterations = 5
        self.elitism_rate = 0.3

    def local_search(self, individual, func, lb, ub):
        best_local = individual
        best_value = func(individual)
        for _ in range(self.local_search_iterations):
            candidate = best_local + 0.1 * np.random.normal(size=self.dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_value = func(candidate)
            if candidate_value < best_value:
                best_local = candidate
                best_value = candidate_value
        return best_local, best_value

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] + 
                               self.c1 * r1 * (personal_best_position[i] - position[i]) + 
                               self.c2 * r2 * (global_best_position - position[i]))
                position[i] += velocity[i]
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

            # Apply local search to a fraction of best individuals
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(personal_best_value)[:elite_count]
            for i in elite_indices:
                if evaluations >= self.budget:
                    break
                new_position, new_value = self.local_search(personal_best_position[i], func, lb, ub)
                evaluations += 1
                if new_value < personal_best_value[i]:
                    personal_best_position[i] = new_position
                    personal_best_value[i] = new_value

                if new_value < global_best_value:
                    global_best_position = new_position
                    global_best_value = new_value

        return global_best_position, global_best_value