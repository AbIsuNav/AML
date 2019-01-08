import pickle
import random
import itertools
import numpy as np
from collections import defaultdict


#Variables you need
M = 6  #30
N = 6  #20
R = 10 # rounds
beta = np.pi
sigma = 4
# mean for each subfield with respect to the player
mu = np.array([np.random.normal(10, sigma, (M,N)), 
               np.random.normal(23, sigma, (M,N))])
print ("Mean matrix shape: ",mu.shape)


#Function to save dictionary in a pickle file
def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def generate_data():            
    '''
    Function to generate tournament data
    '''
    # Generate Players List and list of pairs
    players_list = [i+1 for i in range(N)]
    player_pairs = list(itertools.combinations(players_list, 2))
    output_sequences = defaultdict(dict)

    for player_pair in player_pairs:
        for i in range(R):
            output_sequences[player_pair][i+1] = []
            sequence = []
            table_idx = 0     #If we are in left side of M or right side
            for j in range(M):
                choice = np.random.choice(['switch', 'stay'], p=[3/float(4), 1/float(4)])
                if choice == 'switch':
                    table_idx = 1 - table_idx
                player1_performance = np.random.normal(mu[table_idx,j,player_pair[0]-1], beta, 1)
                player2_performance = np.random.normal(mu[table_idx,j,player_pair[1]-1], beta, 1)
                sequence.append(player1_performance+player2_performance)

            output_sequences[player_pair][i+1] = sequence
    return output_sequences


if __name__ == '__main__':
    
    output_sequences = generate_data()

    print("pair of players:", output_sequences.keys())
    '''Save the dictionary'''
    print(mu)
    save_obj(output_sequences,"sequence_output")