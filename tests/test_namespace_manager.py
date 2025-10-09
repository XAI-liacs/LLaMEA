import pytest
import os
import jsonlines
import time
from llamea.utils import prepare_namespace, clean_local_namespace, _add_builtins_into
from llamea.loggers import ExperimentLogger

def test_prepare_namespace_imports_all():
    # Test 1: All libraries are available.
    test1 = {
        "code": """import numpy as np
import random
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math"]
    }

    soln, potential_issue = prepare_namespace(test1["code"], test1["allowed"])
    assert potential_issue is None
    expected_allowed_list = test1["namespace_keys"]
    _add_builtins_into(expected_allowed_list)
    for key in soln:
        assert key in expected_allowed_list
    
def test_prepare_namespace_returns_generic_objects_for_not_allowed_libraries():
    #Test 2 If a library to be imported is not in the allowed list, test return of generic object.
    test2 = {
        "code": """import numpy as np
import random
import scipy
import sklearn
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math", "scipy", "sklearn"],
        "potential_issue" : "scipy, sklearn are currently not allowed to be imported in this framework."
    }
    name_space, potential_issue = prepare_namespace(test2["code"], test2["allowed"])

    for key, value in name_space.items():
        assert isinstance(value, object)
        if key in test2["allowed"]:
            assert isinstance(value, type.ModuleType)
        assert key in test2["namespace_keys"]
    assert potential_issue == test2["potential_issue"]


def test_clean_local_namespace_generates_purely_local_namespace():
    global_ns = {
        "imported_library_a" : object,
        "imported_library_b" : object,
        "imported_library_c" : object
    }

    # Simulating exec returns local_ns += global_ns
    local_ns = {
        "imported_library_a" : object,
        "imported_library_b" : object,
        "imported_library_c" : object,
        "instantiated_object": object,
        "local_list" : [1, 2, 34, 101]
    }

    local_ns2 = clean_local_namespace(local_ns, global_ns)
    for key in global_ns:
        assert key not in local_ns2
        assert key not in local_ns #Reference semantics.


def test_prepare_namespace_logs_data():
    test1 = {
        "code": """import numpy as np
import random
import scipy
import sklearn
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math", "scipy", "sklearn"],
        "potential_issue" : "scipy, sklearn are currently not allowed to be imported in this framework.",
        "import_failure" : ["scipy", "sklearn"]
    }

    test2 = {
        "code": """import numpy as np
import random
import scipy
import sklearn
import Qiskit.Aer
import shapely
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math", "scipy", "sklearn", "Qiskit", "shapely"],
        "potential_issue" : "scipy, sklearn, Qiskit, shapely are currently not allowed to be imported in this framework.",
        "import_failure" : ["scipy", "sklearn", "Qiskit.Aer", "shapely"]
    }

    logger = ExperimentLogger("test-prepare-namespace")


    potential_issues = []

    
    ns, potential_issue = prepare_namespace(test1["code"], test1["allowed"], logger)
    potential_issues.append(potential_issue)
    for key in ns:
        assert key in test1["namespace_keys"]
    ns, potential_issue = prepare_namespace(test2["code"], test2["allowed"], logger)
    potential_issues.append(potential_issue)
    for key in ns:
        assert key in test2["namespace_keys"]
    
    with jsonlines.open(os.path.join(logger.dirname, "import_failures.jsonl"), "r") as reader:
        count = 0
        logged_objects = []
        for obj in reader:
            logged_objects.append(obj)
            count += 1
    assert count == 2
    for rejected_library in logged_objects[0]["import_misses"]:
        assert rejected_library in test1["import_failure"]
    
    for rejected_library in logged_objects[1]["import_misses"]:
        assert rejected_library in test2["import_failure"]