import numpy as np

class BioAM_ABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_bees = max(10, dim)
        self.limit = self.num_bees * 1.5  # Limit for abandonment
        self.local_search_intensity = 0.1  # Local exploitation intensity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.num_bees, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.num_bees
        trial = np.zeros(self.num_bees)

        # Helper functions
        def local_search(ind):
            direction = np.random.normal(0, 1, self.dim)
            step_size = self.local_search_intensity * (ub - lb)
            new_ind = ind + step_size * direction
            return np.clip(new_ind, lb, ub)

        def random_selection(exclude_idx):
            indices = list(range(self.num_bees))
            indices.remove(exclude_idx)
            return np.random.choice(indices)

        best_solution = population[np.argmin(fitness)]
        best_fitness = fitness.min()

        while evaluations < self.budget:
            # Employed bee phase
            for i in range(self.num_bees):
                phi = np.random.uniform(-1, 1, self.dim)
                partner_idx = random_selection(i)
                new_solution = population[i] + phi * (population[i] - population[partner_idx])
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                evaluations += 1
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

                # Memetic local search
                if np.random.rand() < 0.1:
                    refined_solution = local_search(population[i])
                    refined_fitness = func(refined_solution)
                    evaluations += 1
                    if refined_fitness < fitness[i]:
                        population[i] = refined_solution
                        fitness[i] = refined_fitness

            # Onlooker bee phase
            total_fitness = np.sum(1 / (1 + fitness))
            probabilities = (1 / (1 + fitness)) / total_fitness
            for _ in range(self.num_bees):
                i = np.random.choice(range(self.num_bees), p=probabilities)
                phi = np.random.uniform(-1, 1, self.dim)
                partner_idx = random_selection(i)
                new_solution = population[i] + phi * (population[i] - population[partner_idx])
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                evaluations += 1
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

            # Scout bee phase
            for i in range(self.num_bees):
                if trial[i] > self.limit:
                    population[i] = np.random.uniform(lb, ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1
                    trial[i] = 0

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx]

        return best_solution, best_fitness