import copy
import json
import traceback
import uuid
from typing import Optional

import numpy as np

from llamea.multi_objective_fitness import Fitness, is_multi_objective_fitness


class Solution:
    """
    Represents a candidate solution (an individual) in the evolutionary algorithm.
    Each individual has properties such as code, fitness, feedback, and metadata for additional information.
    """

    def __init__(
        self,
        code="",
        name="",
        description="",
        configspace=None,
        generation=0,
        parent_ids=None,
        operator=None,
        task_prompt="",
    ):
        """
        Initializes an individual with optional attributes.

        Args:
            code (str): The code of the individual.
            name (str): The name of the individual (typically the class name in the code).
            description (str): A short description of the individual (e.g., algorithm's purpose or behavior).
            configspace (Optional[ConfigSpace]): Optional configuration space for HPO.
            generation (int): The generation this individual belongs to.
            parent_ids (list): UUID of the parent individuals in a list.
            operator (str): Optional identifier of the LLM operation that created this individual.
            task_prompt (str): The task prompt used to generate this solution.
        """
        self.id = str(uuid.uuid4())  # Unique ID for this individual
        self.code = code
        self.name = name
        self.description = description
        self.configspace = configspace
        self.generation = generation
        self.fitness = float("nan")
        self.feedback = ""
        self.error = ""
        self.parent_ids = list(parent_ids) if parent_ids is not None else []
        self.metadata = {}  # Dictionary to store additional metadata
        self.operator = operator
        self.operator_id = None  # Operator.id, set by set_operator()
        self.task_prompt = task_prompt

    def __getstate__(self):
        # The attributes themselves, so that a ConfigurationSpace survives
        # pickling (joblib workers, deepcopy, warm start); ``to_dict`` would
        # turn it into a plain dict.
        return self.__dict__.copy()

    def __setstate__(self, state):
        state.setdefault("operator_id", None)  # older pickles lack this field
        state.setdefault("metadata", {})
        self.__dict__.update(state)
        if self.configspace == "":
            self.configspace = None
        elif isinstance(self.configspace, dict):
            # Pickles written before 1.6.0 hold the serialised dict.
            try:
                from ConfigSpace import ConfigurationSpace

                self.configspace = ConfigurationSpace.from_serialized_dict(
                    self.configspace
                )
            except Exception:
                pass

    def set_operator(self, operator):
        """
        Sets the operator name that generated this individual.

        Args:
            operator (tuple): ``(operator, *parent_ids)``: the
                :class:`~llamea.operator.Operator` and the ids of the parents
                used to generate this individual.
        """
        self.operator = f"Operator: {operator[0]}, with parents: {operator[1:]}."
        self.operator_id = operator[0].id

    def add_metadata(self, key, value):
        """
        Adds key-value pairs to the metadata dictionary.

        Args:
            key (str): The key for the metadata.
            value: The value associated with the key.
        """
        self.metadata[key] = value

    def get_metadata(self, key):
        """
        Get a metadata item from the dictionary.

        Args:
            key (str): The key for the metadata to obtain.
        """
        return self.metadata[key] if key in self.metadata.keys() else None

    def set_scores(
        self, fitness: float | Fitness, feedback="", error: Optional[Exception] = None
    ):
        """Sets the score, feedback and (optional) error of this individual.

        Args:
            fitness (float | Fitness): The score; a ``float`` for a single
                objective, a :class:`~llamea.multi_objective_fitness.Fitness`
                for several objectives.
            feedback (str): Feedback for the LLM, e.g. suggested improvements
                or the target score.
            error (Exception | None): The exception raised while running the
                code, if any. Its type, message and the offending line of the
                code are stored in ``error`` and shown to the LLM.
        """
        self.fitness = fitness
        self.feedback = feedback

        if error:
            error_type = type(error).__name__
            error_msg = str(error)

            self.error = f"{error_type}: {error_msg}.\n"

            code_lines = self.code.split("\n") if self.code else []

            line_no = None
            code_line = ""

            if getattr(error, "__traceback__", None) is not None:
                tb = traceback.extract_tb(error.__traceback__)
                # Prefer the innermost frame in the candidate code (exec'd
                # code has the file name "<string>"); the innermost frame
                # overall is often inside numpy or scipy.
                own = [f for f in tb if f.filename.startswith("<")]
                frames = own or tb
                if frames:
                    frame = frames[-1]
                    line_no = frame.lineno

                    if 1 <= line_no <= len(code_lines):
                        code_line = code_lines[line_no - 1]

            if line_no is not None:
                self.error += f"On line {line_no}: {code_line}.\n"

    def get_fitness_vector(self) -> list[float]:
        if is_multi_objective_fitness(self.fitness):
            vector = []
            for key in sorted(self.fitness.keys()):
                vector.append(self.fitness[key])
            return vector
        else:
            return [self.fitness]

    def get_summary(self):
        """
        Returns a string summary of this solution's key attributes.

        Returns:
            str: A string representing the solution in a summary format.
        """
        return f"{self.name}: {self.description} (Score: {self.fitness})"

    def copy(self):
        """
        Returns a copy of this solution, with a new unique ID and a reference to the current solution as its parent.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_solution = Solution(
            code=self.code,
            name=self.name,
            description=self.description,
            configspace=self.configspace,
            generation=self.generation + 1,
            parent_ids=[self.id],  # Link this solution as the parent
            operator=self.operator,
            task_prompt=self.task_prompt,
        )
        new_solution.feedback = self.feedback
        new_solution.error = self.error
        new_solution.metadata = copy.deepcopy(self.metadata)  # nested dicts too
        new_solution.operator_id = self.operator_id
        return new_solution

    def empty_copy(self):
        """
        Returns a copy of this solution, with a new unique ID and a reference to the current solution as its parent but without other fields.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_solution = Solution(
            code="",
            name="",
            description="",
            configspace=None,
            generation=self.generation + 1,
            parent_ids=[self.id],  # Link this solution as the parent
            operator=self.operator,
            task_prompt=self.task_prompt,
        )
        return new_solution

    def to_dict(self):
        """
        Converts the individual to a dictionary.

        Returns:
            dict: A dictionary representation of the individual.
        """
        try:
            cs = self.configspace
            cs = cs.to_serialized_dict()
        except Exception:
            cs = ""
        return {
            "id": self.id,
            "fitness": self.fitness,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "configspace": cs,
            "generation": self.generation,
            "feedback": self.feedback,
            "error": self.error,
            "parent_ids": self.parent_ids,
            "operator": self.operator,
            "operator_id": self.operator_id,
            "metadata": self.metadata,
            "task_prompt": self.task_prompt,
        }

    def to_json(self):
        """
        Converts the individual to a JSON string.

        Returns:
            str: A JSON string representation of the individual.
        """
        return json.dumps(self.to_dict(), default=str, indent=4)
