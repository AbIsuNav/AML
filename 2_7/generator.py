import numpy as np
import random as rd

# GENERATOR FUNCTIONS _____________________________________________________________________________________________________________________________

def define_HMMs(K,R,M):
	# Class probabilities - one class is much more probable than the rest
	pi = np.zeros((K))
	class1 = rd.randint(0,K-1)
	pi[class1] = 0.5
	for k in range(K):
		if pi[k]==0.0:
			pi[k] = 0.5/(K-1)

	# Start probabilities - UNIFORM
	init = 1/R*np.ones((R))

	# Transition probabilities - from each row, there are only two possible next states, with varying probabilities
	A = np.zeros((K, R, R))
	for k in range(K):
		for r in range(R):
			rand = rd.randint(10,20)
			row1 = rd.randint(0,R-1)
			row2 = rd.randint(0,R-1)
			while(row2 == row1):
				row2 = rd.randint(0,R-1)

			A[k,r,row1] = rand/20
			A[k,r,row2] = (20-rand)/20

	# Emission probabilities - different noise for different classes, but same noise for all rows within that class
	E = np.zeros((K, R, R))
	for k in range(K):
		rand = rd.randint(10,20)
		for r in range(R):
			E[k,r,r] = rand/20
			E[k,r,(r+1)%R] = (20-rand)/40
			E[k,r,(r-1)%R] = (20-rand)/40

	return pi, init, A, E


def generate_states(k,R,M,start_prob,transition_prob):
	init = start_prob
	X = np.zeros((M), dtype=int)

	rand = rd.random()
	sum_steps = 0.0
	for r in range(R):
		if rand>=sum_steps and rand<sum_steps+init[r]:
			X[0] = r
			break
		sum_steps += init[r]

	for m in range(1,M):
		A = transition_prob[k,X[m-1],:]
		rand = rd.random()
		sum_steps = 0.0
		for r in range(R):
			if rand>=sum_steps and rand<sum_steps+A[r]:
				X[m] = r;
				break
			sum_steps += A[r]
	
	return X

		
def generate_observations(k,R,M,X,emission_prob):
	Z =  np.zeros((M), dtype=int)
	for m in range(M):
		E = emission_prob[k,X[m],:]
		rand = rd.random()
		sum_steps = 0.0
		for r in range(R):
			if rand>=sum_steps and rand<sum_steps+E[r]:
				Z[m] = r
				break
			sum_steps += E[r]
	
	return Z


def generate_data(N,K,R,M,class_prob,emission_prob,start_prob,transition_prob):
	classes = np.zeros((N), dtype=int)
	observations = np.zeros((N,M), dtype=int)

	for n in range(N):
		rand = rd.random()
		sum_steps = 0.0
		for k in range(K):
			if rand>=sum_steps and rand<sum_steps+class_prob[k]:
				k_n = k;
				break
			sum_steps += class_prob[k]

		classes[n] = k_n
		observations[n,:] = generate_observations(k_n, R, M, generate_states(k_n, R, M,start_prob,transition_prob),emission_prob)

	return classes, observations

# SOLUTION FUNCTIONS _______________________________________________________________________________________________________________________

# Fill in

# _____________________________________________________________________________________________________________________________________

#
# nr_vehicles = 10;  # one row of observations/car
# nr_classes = 5;  # number of emission
# nr_rows = 10;
# nr_columns = 7;  # for observations
# #
# class_prob, start_prob, transition_prob, emission_prob = define_HMMs(nr_classes, nr_rows, nr_columns)
# print("Class probabilities\n", class_prob)
# print("\nStart probabilities\n", start_prob)
# print("\nTransition probabilities\n", transition_prob[0])
# print("\nEmission probabilities\n", emission_prob)
# #
# targets, data = generate_data(nr_vehicles, nr_classes, nr_rows, nr_columns,class_prob,emission_prob,start_prob,transition_prob)
# print("\nObserved sequences\n",data)
# print("\nTrue classes\n", targets)

		
