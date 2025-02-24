import numpy as np
from scipy.optimize import minimize

class HybridPeriodicTabuSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.tabu_list = []

    def adaptive_tabu_search(self, func, bounds, initial_solution=None, max_tabu_size=10):
        lb, ub = bounds.lb, bounds.ub
        if initial_solution is None:
            current_solution = np.random.uniform(lb, ub, self.dim)
        else:
            current_solution = np.copy(initial_solution)

        current_fitness = func(current_solution)
        self.evaluations += 1

        best_solution = np.copy(current_solution)
        best_fitness = current_fitness

        while self.evaluations < self.budget:
            neighbors = self.generate_neighbors(current_solution, bounds)
            best_neighbor, best_neighbor_fitness = None, float('inf')

            for neighbor in neighbors:
                if tuple(neighbor) not in self.tabu_list:
                    fitness = func(neighbor)
                    self.evaluations += 1
                    if fitness < best_neighbor_fitness:
                        best_neighbor_fitness = fitness
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness

            if current_fitness < best_fitness:
                best_solution = np.copy(current_solution)
                best_fitness = current_fitness

            self.tabu_list.append(tuple(current_solution))
            if len(self.tabu_list) > max_tabu_size:
                self.tabu_list.pop(0)

            current_solution = self.enforce_periodicity(current_solution)

        return best_solution, best_fitness

    def generate_neighbors(self, solution, bounds, num_neighbors=10, perturbation=0.1):
        lb, ub = bounds.lb, bounds.ub
        neighbors = []
        for _ in range(num_neighbors):
            neighbor = solution + np.random.uniform(-perturbation, perturbation, self.dim)
            neighbor = np.clip(neighbor, lb, ub)
            neighbors.append(neighbor)
        return neighbors

    def enforce_periodicity(self, solution):
        pattern = self.detect_periodic_pattern(solution)
        if pattern is not None:
            return np.tile(pattern, len(solution) // len(pattern) + 1)[:len(solution)]
        return solution

    def detect_periodic_pattern(self, sequence):
        length = len(sequence)
        autocorrelation = np.correlate(sequence, sequence, mode='full')
        autocorrelation = autocorrelation[length-1:]
        peaks = np.where((autocorrelation[1:] < autocorrelation[:-1]) &
                         (autocorrelation[:-1] > np.mean(autocorrelation) * 1.1))[0]
        if peaks.size > 0:
            period = peaks[0] + 1
            return sequence[:period]
        return None

    def local_search(self, func, x0, bounds):
        std_bounds = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
        result = minimize(func, x0, bounds=std_bounds, method='Nelder-Mead')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.adaptive_tabu_search(func, bounds)
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)
        return best_solution