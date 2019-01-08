import pickle
import numpy as np
import generator as Gen


def get_transition(r1, r2): # constant
    if r1 == r2:
        return 0.25
    else:
        return 0.75


def get_emission(r, m,mu_, obs,sigma): #for one obs and one mu_vector
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (obs - mu_[m][r])**2 / (2 * sigma**2))


def get_init(): #constant
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
    alpha = np.zeros([M, R])

    # base case
    O = []
    for r in range(R):  # for each round 2
        O.append(get_emission(r,0,mu, observations[0], Sigma))
    alpha[0, :] = pi * O[:]

    # recursive case
    for m in range(1, M):
        for r2 in range(R):
            for r1 in range(R):
                transition = get_transition(r1, r2)
                emission = get_emission(r2,m,mu, observations[m],Sigma)
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
                emission = get_emission(r2,m+1,mu, observations[m + 1], Sigma)
                beta[m, r1] += beta[m + 1, r2] * transition * emission

    O = []
    for r in range(R):
        O.append(get_emission(r,0, mu, observations[0], Sigma))

    return beta, np.sum(pi * O[:] * beta[0, :])


def get_gamma(mu, sigma, observations):
    # mu matrix, one sigma, one set M of observations
    R = 2
    T = len(observations) # M
    alpha, a = forward(get_init, get_transition, get_emission, observations, mu, sigma)
    beta, b = backward(get_init, get_transition, get_emission, observations,mu, sigma)
    xi = np.zeros([T,R,R])
    gamma = np.zeros([T,R])
    for t in range(T):
        for i in range(R):
            gamma[t][i] = alpha[t][i]*beta[t][i]/a
    for i in range(R):
        xi[T-1][i] = alpha[T-1][i]
    return gamma, xi, a


def estimate_mu(mu_old, observations,sigma ):
    # returns new mu estaimation for pair of players and all gamas for all rounds for C
    mu_new = np.zeros([Gen.M, 2])  # M*K
    gamas = []  # size Rounds
    xis = [] #size Rounds
    a_sum = []  #size Rounds
    for r in range(Gen.R):  # iterate over rounds
        g, xi, a = get_gamma(mu_old, sigma, observations[r])
        a_sum.append(a)
        gamas.append(g)
        xis.append(xi)
    for k in range(2):
        for n in range(len(observations[0])):
            nom = 0.0
            denom = 0.0
            for r in range(len(observations)):
                #print("GAMA:", gamas[r][n][k])
                nom += gamas[r][n][k]*observations[r][n]
                denom += gamas[r][n][k]
                #print("NOM:", nom)
                #print("Denom:", denom)
            mu_new[n][k] = nom/denom
    return mu_new, gamas, a_sum


def get_Q(mu_old, observations,sigma):
    """
    this returns the new mu and the calculation of Q
    :param mu_old: vector of mu
    :param observations:
    :param sigma:
    :return:
    """

    new_mu, gammas, a_sum = estimate_mu(mu_old, observations,sigma)
    new_Q = 0
    for r in range(len(gammas)):
        for m in range(len(new_mu)): #M
            for k in range(len(new_mu[0])): # K = 2
                new_Q += gammas[r][m][k] * np.log(a_sum[r])
    return new_Q, new_mu, gammas

def estimate_SIGMA(gammas, observations, mus):
    sigma = 0.0
    ga = 0
    for r in range(len(observations)):
        for n in range(len(observations[0])):
            for k in range(2): #states
                sigma += gammas[r][n][k]*(observations[r][n]-mus[n][k])**2
                ga += gammas[r][n][k]
    return np.sqrt(sigma/(ga))



def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


def start_data():
    R = Gen.R
    M = Gen.M
    mu = 20*np.random.rand(M,R)  # initial guess M*R
    sigma = 2
    dictionary_mu = dict()
    dictionary_sigma = dict()
    old_Q, new_Q = 0, 1
    output_sequences = Gen.generate_data()
    for i in output_sequences.keys():  # i pair of players
        observations_player = []
        for r in range(1, R + 1):  # r rounds
            ob = []
            for m in range(M):  # m len of the observations
                ob.append(output_sequences[i][r][m][0])  # this are the observation
            observations_player.append(ob)
        # RUN code for pair of players:
        cont = 0
        while new_Q > old_Q and cont < 100:
            old_Q, new_mu, gammas = get_Q(mu, observations_player,sigma)
            dictionary_mu[i] = new_mu

            print(estimate_SIGMA(gammas,observations_player, new_mu))
            dictionary_sigma[i] = estimate_SIGMA(gammas,observations_player, new_mu)
            sigma = dictionary_sigma[i]
            new_Q, new_mu, gammas = get_Q(new_mu, observations_player, sigma)
            mu = new_mu
            cont+=1
            #print("QS:", old_Q, new_Q)
        old_Q, new_Q = 0, 1
    print("Predicted combined MU: ",dictionary_mu)
    N = Gen.N
    print("pair of players:", output_sequences.keys())

    # SOLVING SYSTEM OF EQ:
    final_mu = np.zeros([2, M, N])
    X = np.zeros([6,N])
    res = np.zeros([6])
    cont = 0
    for k in range(2):
        for m in range(M):
            for p1 in range(1,2):
                for p2 in range(p1 + 1, N + 1):
                    X[cont][p1-1] = 1
                    X[cont][p2-1] = 1
                    res[cont] = dictionary_mu[(p1,p2)][m][k]
                    cont += 1
                X[5][1] = 1
                X[5][2] = 1
                res[cont] = dictionary_mu[(2, 3)][m][k]
            cont = 0
            final_mu[k][m]=np.abs(np.linalg.solve(X,res))


    print("Pair of players:", output_sequences.keys())
    '''Save the dictionary'''
    print(Gen.mu)
    print("Predicted Mu: \\", final_mu)
    Gen.save_obj(output_sequences, "sequence_output")
    print("Error: \\",((Gen.mu - final_mu)**2).mean(axis=0))
    print("Predicted Sigma: \\", dictionary_sigma)
#--MAIN:
start_data()