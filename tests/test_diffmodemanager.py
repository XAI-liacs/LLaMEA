# test_diff_mode_manager.py
from llamea.diffmodemanager import DiffModeManager


# -----------------------------
# Helpers
# -----------------------------

def run_dm(old_code, diff_block):
    dm = DiffModeManager(old_code, diff_block)
    return dm()


def test_perfect_match():
    old = """class A:
    def f(self):
        return 1"""

    diff = """<<<<<<< SEARCH
        return 1
=======
        return 2
>>>>>>> REPLACE"""

    result = run_dm(old, diff)
    assert "return 2" in result
    assert "return 1" not in result


def test_whitespace_mismatch():
    old = """class A:
    def f(self):
        return 1"""

    diff = """<<<<<<< SEARCH
        return 1 
=======
        return 42
>>>>>>> REPLACE"""

    result = run_dm(old, diff)
    print(result)
    # strict matcher should fail
    assert "return 42" in result
    assert "return 1" not in result

def test_malformed_missing_search():
    old = "x = 1"

    diff = """=======
x = 2
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Should safely no-op
    assert result == old


def test_malformed_duplicate_search():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Should not crash, may or may not apply
    assert isinstance(result, str)
    assert old == result


def test_empty_search_block():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
=======
x = 2
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Malformed blocks must not mutate.
    assert isinstance(result, str)
    assert result == old


def test_empty_replace_block():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
x = 1
=======
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Test Deletion Patch applied.
    assert isinstance(result, str)
    assert result == ""


def test_extra_spacing():
    old = """x = 1
y = 2"""

    diff = """<<<<<<< SEARCH
x = 1

=======
x = 100

>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Invalid search, must not update the code base.
    assert isinstance(result, str)
    assert old == result


def test_multiple_mutations():
    old = """x = 1
y = 2"""

    diff = """<<<<<<< SEARCH
x = 1
=======
x = 10
>>>>>>> REPLACE
<<<<<<< SEARCH
y = 2
=======
y = 20
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    assert "x = 10" in result
    assert "y = 20" in result


def test_no_match():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
z = 9
=======
z = 10
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    assert result == old


def test_partial_match():
    old = """def f():
    return a - b"""

    diff = """<<<<<<< SEARCH
    return a -
=======
    return a + b
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # strict matcher should fail
    assert old == result


def test_indentation_drift():
    old = """def f():
    return a - b"""

    diff = """<<<<<<< SEARCH
return a - b
=======
    return a + b
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    assert "return a + b" not in result


def test_broken_markers():
    old = "x = 1"

    diff = """<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLAC"""

    result = run_dm(old, diff)

    # Shouldn't trigger generate mutations.append(*)
    assert old == result


def test_conflicting_mutations():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE
<<<<<<< SEARCH
x = 1
=======
x = 3
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    # Applying second mutation at same line count as already implemented, must fail.
    assert "x = 2" in result and "x = 3" not in result


def test_extra_separators():
    old = "x = 1"

    diff = """<<<<<<< SEARCH
x = 1
=======
=======
x = 2
>>>>>>> REPLACE"""

    result = run_dm(old, diff)

    assert isinstance(result, str)
    assert result == old


def test_m_to_n_mutations():
    old= """
def Sum(upto: int) -> int:
    sum = 0
    for i in range(upto + 1):
        sum += i
    return sum
"""

    diff="""<<<<<< SEARCH
    sum = 0   
    for i in range(upto + 1):
        sum += i  
    return sum
==========
    return (n + 1) * n // 2
>>>>>>REPLACE
"""

    result = run_dm(old, diff)

    assert result == """
def Sum(upto: int) -> int:
    return (n + 1) * n // 2
"""

def test_diff_mode_ignores_wrong_markers():
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
<<<<<<< SEARCH
>>>>>>> REPLACE
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
>>>>>>> REPLACE"""

    result = run_dm(code, response)

    new_code = """class Calculator:
    def add(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a + b

    def subtract(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a + b

    def multiply(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        return a / b

    def divide(self, a, b):
        assert isinstance(a, (float, int))
        assert isinstance(b, (float, int))
        assert b != 0
        return a / b
"""
    print(result)
    print('-----------------------')
    print(new_code)
    assert result == new_code

