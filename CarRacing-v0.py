import matplotlib.pyplot as plt
import gym
import os
import time
import random
import time
import numpy as np
from skimage import transform
from IPython.display import display, clear_output
import tensorflow as tf
from utils import FrameStack, Scheduler, calculate_expected_return
from ppo import PPO
from vec_env.subproc_vec_env import SubprocVecEnv

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    frame = frame * 2 - 1
    return frame

def make_env():
    return gym.make("CarRacing-v0")

def evaluate(test_env, num_steps=None):
    initial_frame = test_env.reset()
    frame_stack = FrameStack(initial_frame, preprocess_fn=preprocess_frame)
    done = False
    step = 0
    while not done:
        if num_steps is not None and step > num_steps: break
        step += 1
        # Predict action given state: π(a_t | s_t; θ)
        state = frame_stack.get_state()
        action, _ = model.predict(np.expand_dims(state, axis=0))
        clear_output(wait=True)
        #print("Mean:",action_mean,"Std:",action_std)
        #action = np.random.normal(loc=actions_mean[0], scale=actions_std[0])
        frame, reward, done, info = test_env.step(action[0])
        test_env.render()
        frame_stack.add_frame(frame)
        time.sleep(0.016)

def main():
    # Create environments
    num_envs = 4
    envs = SubprocVecEnv([make_env for _ in range(num_envs)])
    test_env = gym.make("CarRacing-v0")

    # Traning parameters
    lr_scheduler     = Scheduler(initial_value=3e-4, interval=100, decay_factor=0.75)
    discount_factor  = 0.99
    gae_lambda       = 0.95
    ppo_epsilon      = 0.2
    t_max            = 180
    num_epochs       = 10
    batch_size       = 64
    save_interval    = 50

    # Environment constants
    frame_stack_size = 4
    input_shape      = (84, 84, frame_stack_size)
    num_actions      = envs.action_space.shape[0]
    action_min = np.array([-1.0, 0.0, 0.0])
    action_max = np.array([ 1.0, 1.0, 1.0])
    frame_skip = 100

    # Create model
    model_checkpoint = None
    model = PPO(num_actions, input_shape, action_min, action_max, ppo_epsilon,
                value_scale=0.5, entropy_scale=0.001,
                model_checkpoint=model_checkpoint, model_name="CarRacing-v0")

    episode = 0
    while True:
        print("Resetting environments...")
        episode += 1
        T = 0
        
        # Reset environments and get initial frame
        envs.reset()
        envs.get_images()
        for _ in range(frame_skip):
            envs.step_async(np.zeros((num_envs, num_actions)))
            initial_frames, rewards, dones, infos = envs.step_wait()
        frame_stacks = [FrameStack(initial_frames[i], preprocess_fn=preprocess_frame) for i in range(num_envs)]
        learning_rate = np.maximum(lr_scheduler.get_value(), 1e-6)
        total_reward = 0
        episode_loss = episode_policy_loss = episode_value_loss = episode_entropy_loss = 0
        episode_epochs = 0
        
        # While there are running environments
        print("Training...")
        while T < 1000 - frame_skip:
            states_buffer, taken_actions_buffer, values_buffer, returns_buffer, rewards_buffer = [], [], [], [], []
            
            # Simulate game for some number of steps
            for _ in range(t_max):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                states = [frame_stacks[i].get_state() if dones[i] == False else np.zeros(input_shape) for i in range(num_envs)]
                actions, values = model.predict(states, use_old_policy=True)
                
                # Sample action from a Gaussian distribution
                envs.step_async(actions)
                frames, rewards, dones, infos = envs.step_wait()
                rewards = np.array(rewards)
                envs.get_images() # render
                
                # Store state, action and reward
                states_buffer.append(states)
                taken_actions_buffer.append(actions)
                values_buffer.append(values)
                rewards_buffer.append(rewards)
                total_reward += np.sum(rewards)
                
                # Get new state
                for i in range(num_envs):
                    frame_stacks[i].add_frame(frames[i])
                    
                T += 1

            # Calculate return (discounted rewards over a trajectory)
            states = [frame_stacks[i].get_state() if dones[i] == False else np.zeros(input_shape) for i in range(num_envs)]
            last_values = model.predict(states)[-1]
            rewards_buffer = np.array(rewards_buffer)
            for i in range(num_envs):
                if dones[i] == False:
                    returns_buffer.append(calculate_expected_return(np.append(rewards_buffer[:, i], last_values[i]), discount_factor)[:-1])
                else:
                    returns_buffer.append(calculate_expected_return(np.append(rewards_buffer[:, i], 0), discount_factor)[:-1])
    
            states_buffer        = np.array(states_buffer).reshape((-1, *input_shape))
            taken_actions_buffer = np.array(taken_actions_buffer).reshape((-1, num_actions))
            returns_buffer       = np.array(returns_buffer).transpose(1, 0).flatten()
            advantages_buffer    = returns_buffer - np.array(values_buffer).flatten()
            
            # Train for some number of epochs
            for epoch in range(num_epochs):
                # Sample mini-batch randomly and train
                mb_idx = np.random.choice(len(states_buffer), batch_size, replace=False)

                # Optimize network
                eploss, pgloss, vloss, entloss = model.train(states_buffer[mb_idx], taken_actions_buffer[mb_idx],
                                                             returns_buffer[mb_idx], advantages_buffer[mb_idx],
                                                             learning_rate=learning_rate)
                episode_loss         += eploss
                episode_policy_loss  += pgloss
                episode_value_loss   += vloss
                episode_entropy_loss += entloss
                episode_epochs += 1
            model.update_old_policy() # θ_old <- θ
        average_episode_reward = total_reward / num_envs
        
        episode_policy_loss  /= episode_epochs
        episode_value_loss   /= episode_epochs
        episode_entropy_loss /= episode_epochs
        episode_loss         /= episode_epochs

        clear_output(wait=True)
        print("-- Episode {} --".format(episode))
        print("Learning rate:", learning_rate)
        print("Episode policy loss:", episode_policy_loss)
        print("Episode value loss:", episode_value_loss)
        print("Episode entropy loss:", episode_entropy_loss)
        print("Episode loss:", episode_loss)
        print("Average episode reward:", average_episode_reward)
        print("")
        model.write_episode_summary(episode_policy_loss, episode_value_loss,
                                    episode_entropy_loss, episode_loss,
                                    average_episode_reward, learning_rate)
        if episode % save_interval == 0:
            model.save()

    print("Done!")

    # evaluate(test_env)

if __name__ == "__main__":
    main()