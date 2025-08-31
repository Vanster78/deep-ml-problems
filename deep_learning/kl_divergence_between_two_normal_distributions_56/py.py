import math

def solve(mu_p, sigma_p, mu_q, sigma_q):
	return math.log(sigma_q / sigma_p) + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 0.5

kl_divergence_normal = solve
