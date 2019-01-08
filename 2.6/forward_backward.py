import numpy as np


def get_transition(r1, r2):
    # calculate A((r1, m1), (r2, m1+1)) (for test purpose we set below)
    if r1 == r2:
        return 0.25
    else:
        return 0.75


def get_emission(r, t, mu_vector, sigma, obs):
    # calculate O(m, o) (for test purpose we set below)
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (obs - mu_vector[r][t])**2 / (2 * sigma**2))


def get_init():
    # provide an array containing the initial state probability having size R (for test purpose we set below)
    pi = np.array([0.25,0.75])
    # pi = np.array([0.2, 0.8])
    # number of rows
    R = pi.shape[0]
    return pi, R


def forward(get_init, get_transition, get_emission, observations, mu, Sigma):
    # observations : M*1
    pi, R = get_init()
    M = len(observations)
    alpha = np.zeros((M, R))

    # base case
    O = []
    for r in range(R):  # for each round
        O.append(get_emission(r, observations[0], mu, Sigma))
    alpha[0, :] = pi * O[:]

    # recursive case
    for m in range(1, M):
        for r2 in range(R):
            for r1 in range(R):
                transition = get_transition(r1, r2)
                emission = get_emission(r2, observations[m])
                alpha[m, r2] += alpha[m - 1, r1] * transition * emission

    return alpha, np.sum(alpha[M - 1, :])


def backward(get_init, get_transition, get_emission, observations, mu, Sigma):
    pi, R = get_init()
    M = len(observations)
    beta = np.zeros((M, R))

    # base case
    beta[M - 1, :] = 1

    # recursive case
    for m in range(M - 2, -1, -1):
        for r1 in range(R):
            for r2 in range(R):
                transition = get_transition(r1, r2)
                emission = get_emission(r2, observations[m + 1], mu, Sigma)
                beta[m, r1] += beta[m + 1, r2] * transition * emission

    O = []
    for r in range(R):
        O.append(get_emission(r,observations[0]))

    return beta, np.sum(pi * O[:] * beta[0, :])


def get_gamma(mu, Sigma, R, observations):

    alpha, a = forward(get_init, get_transition, get_emission, observations, mu, Sigma)
    beta, b = backward(get_init, get_transition, get_emission, observations,mu, Sigma)

    Gamma = np.zeros((len(observations), R))
    likelihood = 0.0
    for n in range(len(observations)):
        likelihood += sum(alpha[n] * beta[n])

    for n in range(len(observations)):
        for k in range(len(R)):
            Gamma[n][k] = alpha[n][k] * beta[n][k] / likelihood
    return Gamma


# main :
A_mat = [[1/4,3/4],[1/4,3/4]]
Pi_mat = [1/4,3/4]
mu = [[]]  # size R*M
Sigma = []  # size 1
obs = [[]]  # size Rounds*M
iterate = 50

for round in range(len(obs)):
    gamma = get_gamma(A_mat, mu, Sigma, 2, obs[round])
    for k in range(2):
        nom, denom = 0.0, 0.0
        for n in range(len(obs[0])):
            nom += gamma[n][k]*obs[r][n]
            denom += gamma[n][k]




# test examples
# print(forward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))
# print(backward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))
