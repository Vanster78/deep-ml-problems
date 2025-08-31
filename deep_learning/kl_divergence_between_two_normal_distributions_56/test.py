test_cases = [
    {
        'input': {
            'mu_p': 0.0,
            'sigma_p': 1.0,
            'mu_q': 1.0,
            'sigma_q': 1.0,
        },
        'expected_output': 0.5,
    },
    {
        'input': {
            'mu_p': 0.0,
            'sigma_p': 1.0,
            'mu_q': 0.0,
            'sigma_q': 2.0,
        },
        'expected_output': 0.3181471805599453,
    },
    {
        'input': {
            'mu_p': 1.0,
            'sigma_p': 1.0,
            'mu_q': 0.0,
            'sigma_q': 2.0,
        },
        'expected_output': 0.4431471805599453,
    },
]