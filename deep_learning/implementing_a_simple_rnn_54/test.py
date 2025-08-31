import numpy as np

test_cases = [
    {
        'input': {
            'input_sequence': np.array([[1.0], [2.0], [3.0]]),
            'initial_hidden_state': np.array([0.0]),
            'Wx': np.array([[0.5]]),
            'Wh': np.array([[0.8]]),
            'b': np.array([0.0]),
        },
        'expected_output': np.array([0.97588162])
    },
]