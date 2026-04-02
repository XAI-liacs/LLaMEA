import re
from typing import Optional
from dataclasses import dataclass


## Author by Ananta Shahane
@dataclass
class Mutation:
    search: list[str]
    replacement: list[str]
    lc: int

    def __repr__(self):
        return (
            f"\n{'\n'.join(self.search)}->{'\n'.join(self.replacement)} @LC : {self.lc}"
        )


class DiffModeManager:
    def __init__(self, old_code: str, diff_block: str):
        """
        DiffModeManager is a custom class that contains some regexes to `Search`; `Separator` and `Replace` Block.
        It is always initialised with a string, `content` which correspods to the code block in the `content`, in LLM written replies.

        ## Args:
        `old_code: str`: Code on which diff mode changes are being applied.
        `diff_block: str`: A series of Search Replace pair block.
        """
        self.content = [line.rstrip() for line in diff_block.split("\n")]
        self.code_block = [line.rstrip() for line in old_code.split("\n")]
        self._index = 0
        self._regex = [
            re.compile(r"<<{3,}\s*SEARCH"),  # Search Regex
            re.compile(r"={3,}"),  # Divider Regex
            re.compile(r">{3,}\s*REPLACE"),  # Replace
        ]

    def _find_sub_list_index(self, sub_list: list):
        n, m = len(sub_list), len(self.code_block)
        for k in range(m - n + 1):
            if self.code_block[k : k + n] == sub_list:
                return k
        return -1

    def get_mutations(self):
        mutations: list[Mutation] = []
        current_state = -1
        search: list[str] = []
        replace: list[str] = []

        i = 0
        while i < len(self.content):
            line = self.content[i]
            # print(line, f"State: {current_state}")

            if self._regex[(current_state + 1) % 3].match(line):
                current_state = (current_state + 1) % 3

                if current_state == 2:
                    self.content.insert(i + 1, "")
            else:
                if any(regex.match(line) is not None for regex in self._regex):
                    current_state = -1
                    search.clear()
                    replace.clear()
                    i += 1
                    continue

                match (current_state):
                    case 0:
                        search.append(line)
                    case 1:
                        replace.append(line)
                    case 2:
                        lc = self._find_sub_list_index(search)
                        if (
                            lc != -1
                            and len(search) != 0
                            and lc not in list(map(lambda x: x.lc, mutations))
                        ):
                            mutations.append(Mutation(search[:], replace[:], lc))
                        search.clear()
                        replace.clear()

            i += 1
        return mutations

    def mutate_code(self):
        for mutation in sorted(self.get_mutations(), key=lambda x: x.lc, reverse=True):
            start = mutation.lc
            end = start + len(mutation.search)
            self.code_block[start:end] = mutation.replacement
        return "\n".join(self.code_block)

    def __call__(self):
        return self.mutate_code()


if __name__ == "__main__":
    code = """class Calculator:
    def add(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a - b
    
    def substract(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a + b
    
    def multiply(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a / b
    
    def divide(self, a, b)
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a * b
"""

    response = """<<<<<<< SEARCH
        return a - b
=======
        return a + b
>>>>>>> REPLACE
<<<<<<< SEARCH
    def substract(self, a, b):
=======
    def subtract(self, a, b):
>>>>>>> REPLACE
<<<<<<< SEARCH
        return a + b
=======
        return a - b
>>>>>>> REPLACE
<<<<<<< SEARCH
        return a / b
=======
        return a * b
>>>>>>> REPLACE
<<<<<<< SEARCH
    def divide(self, a, b)
=======
    def divide(self, a, b):
>>>>>>> REPLACE
<<<<<<< SEARCH
        return a * b
=======
        assert b != 0
        return a / b
>>>>>>> REPLACE
"""

    dfm = DiffModeManager(code, response)
    print(dfm())
    # for diff_object in dfm():
    #     print(diff_object)
    #     print('-------------------------------------------------')

    # for line_count, line in dfm.old_code:
    #     print(f"{line_count}: {line}")
