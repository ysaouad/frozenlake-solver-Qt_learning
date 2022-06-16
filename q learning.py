import gym
import matplotlib.pyplot as plt
import numpy as np
from Qt_agent_epsfactor import Qt_Agent

if __name__=='__main__':
    env = gym.make('FrozenLake-v1')
    agent = Qt_Agent(alpha=0.001, gamma = 0.9, eps_start=1, eps_min=0.01, eps_fact=0.9999995, env = env)
    
    scores = []
    win_pct_list = []
    n_games = 500000
       
    for i in range(n_games):
        
        done = False
        agent.state = env.reset()
        score = 0
        
        while not done:
            action = int(agent.action())
            obs, reward,done,info = env.step(action)
            agent.update(action,reward,obs)
            agent.state = obs
            score += reward
            
        scores.append(score)
        
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)        
            if i % 1000 == 0:
                print('episode', i, 'win pct%.2f' % win_pct, 'epsilon %.2f' % agent.eps)

               
    plt.plot(win_pct_list)
    plt.show()