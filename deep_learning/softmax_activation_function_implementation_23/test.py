test_cases = [
    {
        'input': {
            'scores': [10, 10, 10],
        },
        'expected_output': [0.3333, 0.3333, 0.3333]
    },
    {
        'input': {
            'scores': [-100, -100, 100],
        },
        'expected_output': [0., 0., 1.]
    },
    {
        'input': {
            'scores': [-100, 100, 100],
        },
        'expected_output': [0., 0.5, 0.5]
    },
]
