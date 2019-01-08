import numpy as np


def get_transition(r1, r2):
    # calculate A((r1, m1), (r2, m1+1)) (for test purpose we set below)
    return A[r1][r2]


def get_emission(r, o):
    # calculate O(m, o) (for test purpose we set below)

    return B[r][o]


def get_init():
    # provide an array containing the initial state probability having size R (for test purpose we set below)
    pi = np.array([0.2, 0.8])
    # number of rows
    R = pi.shape[0]
    return pi, R


def forward(get_init, get_transition, get_emission, observations):
    pi, R = get_init()
    M = len(observations)
    alpha = np.zeros((M, R))

    # base case
    O = []
    for r in range(R):
        O.append(get_emission(r, observations[0]))
    alpha[0, :] = pi * O[:]

    # recursive case
    for m in range(1, M):
        for r2 in range(R):
            for r1 in range(R):
                transition = get_transition(r1, m, r2)
                emission = get_emission(r2, observations[m])
                alpha[m, r2] += alpha[m - 1, r1] * transition * emission


    return (alpha, np.sum(alpha[M - 1, :]))


def backward(get_init, get_transition, get_emission, observations):
    pi, R = get_init()
    M = len(observations)
    beta = np.zeros((M, R))

    # base case
    beta[M - 1, :] = 1

    # recursive case
    for m in range(M - 2, -1, -1):
        for r1 in range(R):
            for r2 in range(R):
                transition = get_transition(r1, m, r2)
                emission = get_emission(r2, observations[m + 1])
                beta[m, r1] += beta[m + 1, r2] * transition * emission

    O = []
    for r in range(R):
        O.append(get_emission(r, observations[0]))

    return beta, np.sum(pi * O[:] * beta[0, :])



# test examples
print(forward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))
print(backward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))




