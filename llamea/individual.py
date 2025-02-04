import json
import uuid


class Individual:
    """
    Represents a candidate solution (an individual) in the evolutionary algorithm.
    Each individual has properties such as solution code, fitness, feedback, and metadata for additional information.
    """

    def __init__(
        self,
        solution="",
        name="",
        description="",
        configspace=None,
        generation=0,
        parent_id=None,
    ):
        """
        Initializes an individual with optional attributes.

        Args:
            solution (str): The solution (code) of the individual.
            name (str): The name of the individual (typically the class name in the solution).
            description (str): A short description of the individual (e.g., algorithm's purpose or behavior).
            configspace (Optional[ConfigSpace]): Optional configuration space for HPO.
            generation (int): The generation this individual belongs to.
            parent_id (str): UUID of the parent individual.
        """
        self.id = str(uuid.uuid4())  # Unique ID for this individual
        self.solution = solution
        self.name = name
        self.description = description
        self.configspace = configspace
        self.generation = generation
        self.fitness = None
        self.feedback = ""
        self.error = ""
        self.parent_id = parent_id
        self.metadata = {}  # Dictionary to store additional metadata
        self.mutation_prompt = None

    def set_mutation_prompt(self, mutation_prompt):
        """
        Sets the mutation prompt of this individual.

        Args:
            mutation_prompt (str): The mutation instruction to apply to this individual.
        """
        self.mutation_prompt = mutation_prompt

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
        Returns a string summary of this individual's key attributes.

        Returns:
            str: A string representing the individual in a summary format.
        """
        return f"{self.name}: {self.description} (Score: {self.fitness})"

    def copy(self):
        """
        Returns a copy of this individual, with a new unique ID and a reference to the current individual as its parent.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_individual = Individual(
            solution=self.solution,
            name=self.name,
            description=self.description,
            configspace=self.configspace,
            generation=self.generation + 1,
            parent_id=self.id,  # Link this individual as the parent
        )
        new_individual.metadata = self.metadata.copy()  # Copy the metadata as well
        return new_individual

    def to_dict(self):
        """
        Converts the individual to a dictionary.

        Returns:
            dict: A dictionary representation of the individual.
        """
        try:
            cs = self.configspace
            cs = cs.to_serialized_dict()
        except Exception as e:
            cs = ""
        return {
            "id": self.id,
            "solution": self.solution,
            "name": self.name,
            "description": self.description,
            "configspace": cs,
            "generation": self.generation,
            "fitness": self.fitness,
            "feedback": self.feedback,
            "error": self.error,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "mutation_prompt": self.mutation_prompt,
        }

    def to_json(self):
        """
        Converts the individual to a JSON string.

        Returns:
            str: A JSON string representation of the individual.
        """
        return json.dumps(self.to_dict(), default=str, indent=4)
