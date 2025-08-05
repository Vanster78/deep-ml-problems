import importlib.util
import sys
import os
import argparse
import math

import numpy as np
import torch

def load_module_from_path(module_name, file_path):
    """
    Loads a Python module from a given file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        print(f"Error: Could not find module spec for {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading module {file_path}: {e}")
        return None
    return module

def compare_outputs(actual, expected, rtol=1e-5, atol=1e-8):
    """
    Compares actual and expected outputs, handling various data types
    including floats, lists, tuples, numpy arrays, and torch tensors.

    Args:
        actual: The actual output from the solve function.
        expected: The expected output from the test case.
        rtol (float): Relative tolerance for floating-point comparisons.
        atol (float): Absolute tolerance for floating-point comparisons.

    Returns:
        bool: True if outputs are considered equal, False otherwise.
    """
    # Handle None
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False

    # Handle basic types (numbers, strings, booleans)
    if isinstance(actual, (int, str, bool)) and isinstance(expected, (int, str, bool)):
        return actual == expected
    
    # Handle float comparison with tolerance
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)
    
    # Handle lists and tuples (recursively)
    if (isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple))):
        if len(actual) != len(expected):
            return False
        for a_item, e_item in zip(actual, expected):
            if not compare_outputs(a_item, e_item, rtol, atol):
                return False
        return True
    
    # Handle NumPy arrays
    if np is not None and isinstance(actual, np.ndarray):
        if not isinstance(expected, np.ndarray):
            # Try converting expected to numpy array if it's a list/tuple
            try:
                expected = np.array(expected)
            except TypeError:
                return False # Cannot convert expected to numpy array
        
        if actual.shape != expected.shape:
            return False
        
        if actual.dtype.kind == 'f' or expected.dtype.kind == 'f': # If either is float
            return np.allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            return np.array_equal(actual, expected)

    # Handle PyTorch tensors
    if torch is not None and isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor):
            # Try converting expected to torch tensor if it's a list/tuple/numpy array
            try:
                if np is not None and isinstance(expected, np.ndarray):
                    expected = torch.from_numpy(expected)
                else:
                    expected = torch.tensor(expected)
            except (TypeError, ValueError):
                return False # Cannot convert expected to torch tensor
        
        if actual.shape != expected.shape:
            return False

        if actual.dtype.is_floating_point or expected.dtype.is_floating_point:
            return torch.allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            return torch.equal(actual, expected)
            
    # Fallback for other types or type mismatches
    return actual == expected

def run_tests(solution_folder_path):
    """
    Runs tests for all solve implementations in the given solution folder.
    """
    print(f"--- Running tests for folder: {solution_folder_path} ---")

    # 1. Load test.py to get test cases
    test_file_path = os.path.join(solution_folder_path, 'test.py')
    if not os.path.exists(test_file_path):
        print(f"Error: 'test.py' not found in {solution_folder_path}")
        return

    # Add the solution folder to sys.path temporarily to allow relative imports within test.py if any
    sys.path.insert(0, solution_folder_path)
    try:
        test_module = load_module_from_path('test_cases_module', test_file_path)
        if test_module is None:
            print("Failed to load test.py.")
            return

        if not hasattr(test_module, 'test_cases') or not isinstance(test_module.test_cases, list):
            print("Error: 'test.py' must define a list called 'test_cases'.")
            return
        test_cases = test_module.test_cases
        print(f"Found {len(test_cases)} test cases in test.py.")
    finally:
        sys.path.pop(0) # Remove the path

    # 2. Find and load implementation files
    implementation_files = []
    for filename in os.listdir(solution_folder_path):
        if filename.endswith('.py') and filename != 'test.py':
            implementation_files.append(os.path.join(solution_folder_path, filename))

    if not implementation_files:
        print("No implementation files (e.g., py.py, numpy.py) found.")
        return

    for impl_file_path in implementation_files:
        module_name = os.path.splitext(os.path.basename(impl_file_path))[0]
        print(f"\n--- Testing implementation: {module_name}.py ---")

        # Temporarily add the solution folder to sys.path for the implementation module
        sys.path.insert(0, solution_folder_path)
        try:
            impl_module = load_module_from_path(module_name, impl_file_path)
            if impl_module is None:
                print(f"Skipping {module_name}.py due to loading error.")
                continue

            if not hasattr(impl_module, 'solve') or not callable(impl_module.solve):
                print(f"Error: '{module_name}.py' must define a callable function 'solve'.")
                continue

            all_passed = True
            for i, test_case in enumerate(test_cases):
                input_data = test_case['input']
                expected_output = test_case['expected_output']

                try:
                    actual_output = impl_module.solve(**input_data)
                    if compare_outputs(actual_output, expected_output):
                        print(f"  Test Case {i+1}: PASSED")
                    else:
                        print(f"  Test Case {i+1} (Input: {input_data}): FAILED")
                        print(f"    Expected: {expected_output}")
                        print(f"    Actual:   {actual_output}")
                        all_passed = False
                except Exception as e:
                    print(f"  Test Case {i+1} (Input: {input_data}): ERROR during execution")
                    print(f"    Error: {e}")
                    all_passed = False
            
            if all_passed:
                print(f"All tests PASSED for {module_name}.py!")
            else:
                print(f"Some tests FAILED for {module_name}.py.")

        finally:
            sys.path.pop(0) # Remove the path

def main():
    parser = argparse.ArgumentParser(description="Test solution implementations against defined test cases.")
    parser.add_argument('solution_folder', type=str,
                        help='Path to the folder containing test.py and implementation files (e.g., py.py, numpy.py).')
    args = parser.parse_args()

    if not os.path.isdir(args.solution_folder):
        print(f"Error: Folder '{args.solution_folder}' not found.")
        sys.exit(1)

    run_tests(args.solution_folder)

if __name__ == "__main__":
    main()
