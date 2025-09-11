import os
import re
import ast
import jsonlines

from difflib import SequenceMatcher

class DiffManager:
    def __init__(self, diffToolingMethod : str, location : str):
        """
        A testing class that logs relevant stats of implementing diff mode.

        Args:
            diffToolingMethod: A string "generic" | "openEvolve" to pick of the two
            different prompts to implement diff mode.
            location: Directory location for the experiment logging.
        """
        self._diffToolingMethod = diffToolingMethod
        self.logLocation = os.path.join(location, "diff-chat.jsonl")
        self._location = location


    def get_diff_prompt(self) -> str:
        """
            Adds the required prompt to receive diff mode data from LLM.

        Args:
            optimisation_pompt: Optimisation of given individual.
        """
        if self._diffToolingMethod == "generic":
            return """
---
I have a function that replaces a strings `[lines_to_be_changed]` with `[updated_lines]` in the current code. Only provide the required array to apply said patch
MUST use the following format for the solution:
```
[lines_to_be_changed]
========================
[updated_lines]
```
"""
        else:
            return  """
---
You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:
```
<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE
```

Example of valid diff format:
```
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE
```
"""

    def _code_updater(self, code: str, lines_to_change : list[str], updated_lines: list[str]):
        """Line by line update code, and return the update.
        Args:
            code: Current code in the individual.
            lines_to_change: A list of lines to be changed by the LLM.
            updated_lines: Lines to replace the `lines_to_update`.

        """
        if len(lines_to_change) != len(lines_to_change):
            raise ValueError
        for i in range(len(lines_to_change)):
            code = code.replace(lines_to_change[i], updated_lines[i], 1)        #Update one occurance of lines_to_change, to corresponding change.
        return code

    def _apply_generic_change(self, text:str, base_code:str) -> tuple[str, bool, float]:
        """
        Assuming the LLM follows the intructions properly, following format of response is expected.
        ```(python)? <- (python may appear sometimes.)
        ['line1 \n line2', 'line5', 'line 6']
        ===============
        ['replacingline1', 'replacing line 2', 'replacing line 3']

        Where:               _
            line 1            | <-REPLACED BY replacingline1
            line 2            |
                             -
            line 5           <- replaced by replacingline2
            line 6           <- replaced by replacingline3
        Args:
            text: LLM response.text.
            base_code: Base code to be mutated.
        Return:
            Code: updated code, after applying diff.
            bool: Flag about the success of mutation.
            float: Ratio of change in code.
        """
        text = text.strip("```").strip("python")
        parameters = text.split("========================")
        try:
            parameter1 = ast.literal_eval(parameters[0])
            parameter2 = ast.literal_eval(parameters[1])
            code = self._code_updater(base_code, parameter1, parameter2)

            seq_match = SequenceMatcher(None, code, base_code)
            ratio = seq_match.ratio()

            return code, True, ratio
        except Exception:
            return base_code, False, 0.0

    def _apply_open_evolve(self, text:str, base_code: str) -> tuple[str, bool, float]:
        """
        Assuming the LLM follows the intructions properly, following format of response is expected.
        ```(python)? <- (python may appear sometimes.)
        # A series of followin' series will appear.
        <<<<<<< SEARCH
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        =======
        # Reorder loops for better memory access pattern
        for i in range(m):
            for k in range(n):
                for j in range(p):
                    C[i, j] += A[i, k] * B[k, j]
        >>>>>>> REPLACE
        ```

        Args:
            text: LLM response.text.
            base_code: Base code to be mutated.
        Returns:
            Code: updated code, after applying diff.
            bool: Success of diff mode implementation.
            float: Ratio of code changed.
        """
        outLines = []
        inLines = []
        try:
            pattern = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
            new_code = ""
            matches = pattern.findall(text)
            if len(matches) == 0:
                print("WARNING: LLM didn't adhere to search replace pattern. Try bigger model.")
                print(f"response: {text}")
            for search, replace in matches:
                outLines.append(search)
                inLines.append(replace)

            code = self._code_updater(base_code, outLines, inLines)

            seq_match = SequenceMatcher(None, code, base_code)
            ratio = seq_match.ratio()

            return code, True, ratio

        except Exception:
            return base_code, False, 0.0

    def apply_diff(self, base_code: str, responseText: str) -> str:
        """
        Decodes and updates the code.
        Args:
            base_code: Base code that needs to be edited.
            responseText: Response text from the user.
        """
        if self._diffToolingMethod == "generic":
            new_code, success, delta = self._apply_generic_change(responseText, base_code)

        elif self._diffToolingMethod == "openEvolve":
            new_code, success, delta = self._apply_open_evolve(responseText, base_code)
        else:
            new_code = code

        diffObject = {
            "response": responseText,
            "success" : success,
            "diff": delta
        }

        # Log for diagnostics.
        with jsonlines.open(self.logLocation, "a") as jlf:
            jlf.write(diffObject)

        return new_code
