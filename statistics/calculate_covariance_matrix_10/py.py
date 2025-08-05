def solve(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    m = len(vectors[0])

    result = [[0] * n for _ in range(n)]

    means = [sum(v) / m for v in vectors]

    for i in range(n):
        for j in range(i, n):
            s = 0
            for k in range(m):
                s += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
        
            result[i][j] = result[j][i] = s / (m - 1)
    
    return result

calculate_covariance_matrix = solve
