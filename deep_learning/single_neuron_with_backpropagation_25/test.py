import numpy as np

test_cases = [
    {
        'input': {
            'features': np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
            'labels': np.array([1, 0, 0]),
            'initial_weights': np.array([0.1, -0.2]),
            'initial_bias': 0.0,
            'learning_rate': 0.1,
            'epochs': 2
        },
        'expected_output': (np.array([0.1036, -0.1425]), -0.0167, [0.3033, 0.2942])
    },
]