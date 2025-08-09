test_cases = [
    {
        'input': {
            'features': [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
            'labels': [0, 1, 0],
            'weights': [0.7, -0.4],
            'bias': -0.1,
        },
        'expected_output': ([0.4626, 0.4134, 0.6682], 0.3349)
    },
]