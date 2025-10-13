import os
import time
import jsonlines

from llamea.loggers import ExperimentLogger
from llamea.solution import Solution

soln = Solution(
            code="""
import numpy as np
from scipy.spatial import ConvexHull
import random
import math
from statistics import mean

class HeilbronnConvexRegion_n14:

    def __init__(self, n_points: int, best_known_configuration: list[float] | None = None):
        self.n_points = n_points
        self.best_known_configuration = best_known_configuration
        self.points = None
        if best_known_configuration is not None:
            self.points = np.array(best_known_configuration).reshape((n_points, 2))

    def _area(self, points):
        hull = ConvexHull(points)
        return hull.area

    def _rescale(self, points):
        area = self._area(points)
        scale = math.sqrt(1.0 / area)
        centroid = np.mean(points, axis=0)
        points = (points - centroid) * scale + centroid
        return points

    def _smallest_triangle_area(self, points):
        min_area = float('inf')
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                for k in range(j + 1, self.n_points):
                    area = 0.5 * abs(
                        points[i, 0] * (points[j, 1] - points[k, 1]) +
                        points[j, 0] * (points[k, 1] - points[i, 1]) +
                        points[k, 0] * (points[i, 1] - points[j, 1])
                    )
                    min_area = min(min_area, area)
        return min_area

    def _generate_initial_points(self):
        return np.random.rand(self.n_points, 2)

    def __call__(self):
        if self.points is None:
            self.points = self._generate_initial_points()

        self.points = self._rescale(self.points)

        best_points = self.points.copy()
        best_score = self._smallest_triangle_area(self.points)

        temperature = 1.0
        cooling_rate = 0.995
        tolerance = 1e-12

        for _ in range(100000):
            i = random.randint(0, self.n_points - 1)
            original_point = self.points[i].copy()

            # Perturb the point
            self.points[i] += np.random.normal(0, 0.01, 2)

            # Rescale to maintain area of 1
            self.points = self._rescale(self.points)

            new_score = self._smallest_triangle_area(self.points)

            if new_score > best_score:
                best_score = new_score
                best_points = self.points.copy()
            else:
                # Simulated annealing acceptance probability
                acceptance_probability = math.exp((new_score - best_score) / temperature)
                if random.random() < acceptance_probability:
                    pass  # Accept the worse solution
                else:
                    self.points[i] = original_point  # Revert the change

            temperature *= cooling_rate

            if temperature < tolerance:
                break

        return best_points
""",
        name="HeilbronnConvexRegion_n14",
        generation=4
        )

def test_logger_logs_solution():
    solution = soln.copy()
    solution.set_scores(
        0.02783557145848216,
        "min_triangle_area=0.0278356, best known = 0.0278."
    )
    test = {
        "solution" : solution,
        "output" : [[0.63557142, 0.2780757 ],
                    [0.41179303, 0.6500378 ],
                    [0.71410244, 0.29145972],
                    [0.35526134, 0.29541504],
                    [0.38375729, 0.3046403 ],
                    [0.55842241, 0.7181737 ],
                    [0.53782542, 0.32973947],
                    [0.67912473, 0.6429485 ],
                    [0.59160907, 0.55614398],
                    [0.27554739, 0.71103061],
                    [0.48850259, 0.26512655],
                    [0.79653191, 0.71262127],
                    [0.51986098, 0.77731033],
                    [0.75680552, 0.64723116]
                    ]
    }
    logger = ExperimentLogger("test_soln_logging")
    logger.log_solution(test["solution"], test["output"])
    
    last_element = {}
    time.sleep(1)

    with jsonlines.open(os.path.join(logger.dirname, "solutions.jsonl"), "r") as reader:
        for line in reader:
            last_element = line
    
    assert last_element["id"] == test["solution"].id
    assert last_element["fitness"] == test["solution"].fitness
    assert last_element["output"] == test["output"]

def test_logger_ignores_bad_fitness_solutions():
    # Test solution with missing fitness:
    # soln.set_scores(                                                      #Forgot using setscore before logging.
    #     0.02783557145848216,
    #     "min_triangle_area=0.0278356, best known = 0.0278."
    # )


    test = {
        "solution" : soln.copy(),
        "output" : [[0.63557142, 0.2780757 ],
                    [0.41179303, 0.6500378 ],
                    [0.71410244, 0.29145972],
                    [0.35526134, 0.29541504],
                    [0.38375729, 0.3046403 ],
                    [0.55842241, 0.7181737 ],
                    [0.53782542, 0.32973947],
                    [0.67912473, 0.6429485 ],
                    [0.59160907, 0.55614398],
                    [0.27554739, 0.71103061],
                    [0.48850259, 0.26512655],
                    [0.79653191, 0.71262127],
                    [0.51986098, 0.77731033],
                    [0.75680552, 0.64723116]
                    ]
    }
    logger = ExperimentLogger("test_logger_ignores_bad_soln")
    logger.log_solution(test["solution"], test["output"])
    
    last_element = {}
    time.sleep(1)

    with jsonlines.open(os.path.join(logger.dirname, "solutions.jsonl"), "r") as reader:
        for line in reader:
            last_element = line
    
    assert last_element == {}

def test_logger_ignore_solution_without_code():
    solution = soln.copy()
    solution.code = ""

    solution.set_scores(
        0.02783557145848216,
        "min_triangle_area=0.0278356, best known = 0.0278."
    )

    test = {
        "solution" : soln.copy(),
        "output" : [[0.63557142, 0.2780757 ],
                    [0.41179303, 0.6500378 ],
                    [0.71410244, 0.29145972],
                    [0.35526134, 0.29541504],
                    [0.38375729, 0.3046403 ],
                    [0.55842241, 0.7181737 ],
                    [0.53782542, 0.32973947],
                    [0.67912473, 0.6429485 ],
                    [0.59160907, 0.55614398],
                    [0.27554739, 0.71103061],
                    [0.48850259, 0.26512655],
                    [0.79653191, 0.71262127],
                    [0.51986098, 0.77731033],
                    [0.75680552, 0.64723116]
                    ]
    }
    logger = ExperimentLogger("test_logger_ignores_no_soln")
    logger.log_solution(test["solution"], test["output"])
    
    last_element = {}
    time.sleep(1)

    with jsonlines.open(os.path.join(logger.dirname, "solutions.jsonl"), "r") as reader:
        for line in reader:
            last_element = line
    
    assert last_element == {}

def test_ignores_solution_without_output():
    solution = soln.copy()
    solution.set_scores(
        0.02783557145848216,
        "min_triangle_area=0.0278356, best known = 0.0278."
    )
    test = {
        "solution" : solution,
        "output" : None
    }
    logger = ExperimentLogger("test_logger_ignores_soln_without_output")
    logger.log_solution(test["solution"], test["output"])
    
    last_element = {}
    time.sleep(1)

    with jsonlines.open(os.path.join(logger.dirname, "solutions.jsonl"), "r") as reader:
        for line in reader:
            last_element = line
    
    assert last_element == {}