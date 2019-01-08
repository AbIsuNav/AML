import numpy as np
import generator as G
import random as rn


def likelihood(pi, data, transition_prob, emission_prob):
    # number of rows
    R = len(pi)
    Total_L = 0
    likelihood_HMMS = np.ones((len(emission_prob),len(data)))  # N*K
    for k in range(len(transition_prob)):
        A = transition_prob[k]
        B = emission_prob[k]
        for row in range(len(data)):
            x = forward(data[row], A, B, pi, R)
            if ( x > 0.0 ):
                x=x
            else:
                x= rn.uniform(0.0, 1.0e-10)

            likelihood_HMMS[k][row] = x  #* 1/sum(x)
            Total_L += np.log(likelihood_HMMS[k][row])

    return likelihood_HMMS, Total_L


def forward(observations, A, B, pi, R):

    M = len(observations)
    alpha = np.zeros((M, R))

    # base case
    O = []
    for r in range(R):
        O.append(B[r][observations[0]])
    alpha[0, :] = pi * O[:]

    # recursive case
    for m in range(1, M):
        for r2 in range(R):
            for r1 in range(R):
                transition = A[r1][r2]
                emission = B[r2][observations[m]]
                alpha[m, r2] += alpha[m - 1, r1] * transition * emission


    return  np.sum(alpha[M - 1, :]) #alpha,


def responsibilities(pi, likelihoods):
    gamma = np.zeros((len(likelihoods),len(likelihoods[0])))  # K*N

    for k in range(len(likelihoods)):
        for n in range(len(likelihoods[0])):
            gamma[k][n] = pi[k]*likelihoods[k][n]/sum(pi*[likelihoods[i][n] for i in range(len(likelihoods))])
    return gamma


def main():
    nr_vehicles = 70  # one row of observations/car , N
    nr_classes = 5  # number of emission, K
    nr_rows = 10
    nr_columns = 10 # for observations
    class_prob, start_prob, transition_prob, emission_prob = G.define_HMMs(nr_classes, nr_rows, nr_columns)
    targets, data = G.generate_data(nr_vehicles, nr_classes, nr_rows, nr_columns,class_prob,emission_prob,start_prob,transition_prob)
    pi = np.random.dirichlet(np.ones(nr_classes))
    print("initial pi:", pi)
    new_L, old_L = 1,0
    while(new_L>old_L):
        likelihoods, sumLogLikelihood = likelihood(start_prob, data, transition_prob, emission_prob)
        gamma = responsibilities(pi, likelihoods)
        sum2 = 0
        for k in range(nr_classes):
            sum2 += sum(gamma[k]*np.log(pi[k]))
        old_L = sum2 + sumLogLikelihood
        for k in range(nr_classes):
            pi[k] = sum(gamma[k]) / nr_vehicles
        #print(pi)
        gamma = responsibilities(pi, likelihoods)
        sum2 = 0

        for k in range(nr_classes):
            sum2 += sum(gamma[k]*np.log(pi[k]))
        new_L = sum2 + sumLogLikelihood
        print("old_Q ", old_L, " new_Q: ", new_L)
    guess = np.argmax(gamma, axis=0)
    # gamma = responsibilities(pi, likelihoods)
    print("guess:  ", guess)
    print("targets ",targets)
    error=0
    for i in range(len(targets)):
        if targets[i] != guess[i]:
            error+=1
    print("Error: ", error/(len(targets)))
    print("Predicted pi", pi)
    print("Class probabilities", class_prob)
    print("Error pi: ", ((class_prob-pi)**2).mean(axis=0)*100)

if __name__ == "__main__":
    main()