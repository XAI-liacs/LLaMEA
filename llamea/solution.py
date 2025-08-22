import json
import uuid

import numpy as np


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
        parent_ids=[],
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
        self.fitness = -np.inf
        self.feedback = ""
        self.error = ""
        self.parent_ids = parent_ids
        self.metadata = {}  # Dictionary to store additional metadata
        self.operator = operator
        self.task_prompt = task_prompt

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.configspace == "":
            self.configspace = None

    def set_operator(self, operator):
        """
        Sets the operator name that generated this individual.

        Args:
            operator (str): The name of the operator (for logging purposes).
        """
        self.operator = operator

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

    def set_scores(self, fitness, feedback="", error=""):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error

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
        new_solution.metadata = self.metadata.copy()  # Copy the metadata as well
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
