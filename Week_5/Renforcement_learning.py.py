import gym
import random
import numpy
import time

def main():

    env = gym.make("Taxi-v3")

    alpha = 0.9
    gamma = 0.9
    num_of_episodes = 1000
    num_of_steps = 500

    Q_reward = -10*numpy.ones((500,6))

    #print(Q_reward)
    scores = numpy.zeros(10)
    actions_array = numpy.zeros(10)

    


    for i in range(num_of_episodes):
        state1 = env.reset()
        for j in range(num_of_steps):
            random_num = random.randint(0,5)
            state2, reward, done, info = env.step(random_num)
            Q_reward[state1, random_num] = Q_reward[state1, random_num] + alpha*(reward + gamma*max(Q_reward[state2, :]) - Q_reward[state1, random_num])
            state1 = state2

            if done:
                break
    for k in range(10):

        state = env.reset()
        tot_reward = 0
        num_of_actions = 0
        for t in range(50):
            num_of_actions += 1
            action = numpy.argmax(Q_reward[state,:])
            state, reward, done, info = env.step(action)
            tot_reward += reward
            env.render()
            time.sleep(1)
            if done:
                print("Total reward %d" %tot_reward)
                break

        scores[k] = tot_reward
        actions_array[k] = num_of_actions

    print("Average score is: ", numpy.mean(scores))
    print("Average amount of actions is: ", numpy.mean(actions_array))

main()






#env = gym.make("Taxi-v3")
#env.reset()
#env.render()