#Epsilon_Greedy Method Implementation to solve Multi Arm Bandit

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import *

#Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
#It is known optimal choice is Ad Indexed 4
num_bandits = np.shape(dataset)[1]
num_trials = np.shape(dataset)[0]

total_reward = 0
regret = 0


reward_bandits = [0]*num_bandits
num_chosen_bandits = [0]*num_bandits
mean_reward = [0]*num_bandits

chosen_array = []

epsilon = 0.20

for trial in range(0,num_trials) : 
    
    # Decide - Exploration or Exploitation
    choice = np.random.uniform(0,1)
    #print(choice)
    
    if choice < epsilon :
        chosen_bandit = floor(np.random.uniform(0,num_bandits)) 

    else :
        chosen_bandit = np.argmax(mean_reward)
    
    num_chosen_bandits[chosen_bandit]+=1
    reward = dataset.values[trial,chosen_bandit]
    total_reward+=reward
    reward_bandits[chosen_bandit]+=reward
    reward_bandits[chosen_bandit]+=reward
    mean_reward[chosen_bandit] = reward_bandits[chosen_bandit]/num_chosen_bandits[chosen_bandit]
    chosen_array.append(chosen_bandit)
    
    regret+=(dataset.values[trial,4]-reward)


optimal_bandit = np.argmax(num_chosen_bandits)

print("Optimal bandit is "+str(optimal_bandit))
print("Regret" + str(regret) +"Total Reward" + str(total_reward))

plt.hist(num_chosen_bandits)
