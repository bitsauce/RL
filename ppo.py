import tensorflow as tf
import numpy as np
import re
import os

class PolicyGraph():
    def __init__(self, input_states, taken_actions, num_actions, action_min, action_max, scope_name,
                 initial_mean_factor=0.1, initial_std=1.0, clip_action_space=True):
        with tf.variable_scope(scope_name):
            # Construct model
            self.conv1           = tf.layers.conv2d(input_states, filters=16, kernel_size=8, strides=4, activation=tf.nn.relu, padding="valid", name="conv1")
            self.conv2           = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding="valid", name="conv2")
            self.shared_features = tf.layers.flatten(self.conv2, name="flatten")
            
            # Policy branch π(a_t | s_t; θ)
            self.action_mean = tf.layers.dense(self.shared_features, num_actions,
                                               activation=tf.nn.tanh,
                                               kernel_initializer=tf.initializers.variance_scaling(scale=initial_mean_factor),
                                               name="action_mean")
            self.action_mean = action_min + ((self.action_mean + 1) / 2) * (action_max - action_min)
            self.action_std  = tf.layers.dense(self.shared_features, num_actions,
                                               activation=tf.nn.tanh,#activation=tf.nn.softplus,
                                               kernel_initializer=tf.initializers.variance_scaling(scale=initial_std),#tf.zeros_initializer(),#tf.constant_initializer(np.log(np.exp(initial_std) - 1)),
                                               name="action_std") # SoftPlus(x) = log(1 + exp(x))
            self.action_std  = ((self.action_std + 1) / 2) * (action_max - action_min)

            # Failsafe in case std = 0
            self.action_std  = tf.maximum(self.action_std, 1e-6)
            #self.action_mean = tf.check_numerics(self.action_mean, "action_mean")
            #self.action_std = tf.check_numerics(self.action_std, "action_std")

            # Value branch V(s_t; θ)
            self.value = tf.layers.dense(self.shared_features, 1, activation=None, name="value")
        
            # Create graph for sampling actions
            self.action_normal  = tf.distributions.Normal(self.action_mean, self.action_std)
            self.sampled_action = tf.squeeze(self.action_normal.sample(1), axis=0)
            if clip_action_space:
                num_envs   = tf.shape(self.sampled_action)[0]
                action_min = tf.reshape(tf.tile(tf.convert_to_tensor(action_min, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                action_max = tf.reshape(tf.tile(tf.convert_to_tensor(action_max, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                self.sampled_action = tf.clip_by_value(self.sampled_action, action_min, action_max)
            
            # Get the log probability of taken actions
            # log π(a_t | s_t; θ)
            self.action_log_prob = self.action_normal.log_prob(taken_actions)

class PPO():
    def __init__(self, num_actions, input_shape, action_min, action_max, epsilon=0.2, value_scale=0.5, entropy_scale=0.01, model_checkpoint=None, model_name="ppo"):
        tf.reset_default_graph()
        
        self.input_states  = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
        self.taken_actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")
        self.policy        = PolicyGraph(self.input_states, self.taken_actions, num_actions, action_min, action_max, "policy")
        self.policy_old    = PolicyGraph(self.input_states, self.taken_actions, num_actions, action_min, action_max, "policy_old")

        # Create policy gradient train function
        self.returns   = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")
        
        # Calculate ratio:
        # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
        # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
        # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
        self.prob_ratio = tf.exp(self.policy.action_log_prob - self.policy_old.action_log_prob)
        #self.adv = tf.stop_gradient(tf.subtract(tf.expand_dims(self.returns_placeholder, axis=-1), self.value, name="advantage"))
        adv = tf.expand_dims(self.advantage, axis=-1)
        self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv, tf.clip_by_value(self.prob_ratio, 1-epsilon, 1+epsilon) * adv))

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.policy.value), self.returns)) * value_scale
        
        # Entropy loss
        self.entropy_loss = tf.reduce_mean(self.policy.action_normal.entropy()) * entropy_scale
        
        # Total loss
        self.loss = -self.policy_loss + self.value_loss - self.entropy_loss
        
        # Policy parameters
        policy_params      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
        policy_old_params  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
        assert(len(policy_params) == len(policy_old_params))
        for src, dst in zip(policy_params, policy_old_params):
            assert(src.shape == dst.shape)

        # Minimize loss
        self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="lr_placeholder")
        self.optimizer     = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step    = self.optimizer.minimize(self.loss, var_list=policy_params)

        # Update network parameters
        self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

        # Create session
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(tf.global_variables_initializer())
        
        # Episode summary
        tf.summary.scalar("policy_loss", self.policy_loss)
        tf.summary.scalar("value_loss", self.value_loss)
        tf.summary.scalar("entropy_loss", self.entropy_loss)
        tf.summary.scalar("loss", self.loss)
        for i in range(num_actions):
            tf.summary.scalar("taken_actions_{}".format(i), tf.reduce_mean(self.taken_actions[:, i]))
            tf.summary.scalar("prob_ratio_{}".format(i), tf.reduce_mean(self.prob_ratio[:, i]))
            tf.summary.scalar("policy.action_mean_{}".format(i), tf.reduce_mean(self.policy.action_mean[:, i]))
            tf.summary.scalar("policy.action_std_{}".format(i), tf.reduce_mean(self.policy.action_std[:, i]))
        tf.summary.scalar("returns", tf.reduce_mean(self.returns))
        tf.summary.scalar("advantage", tf.reduce_mean(self.advantage))
        self.summary_merged = tf.summary.merge_all()
        
        # Load model checkpoint if provided
        self.model_name = model_name
        self.saver = tf.train.Saver()
        if model_checkpoint:
            self.run_idx = int(re.findall(r"/run\d+", model_checkpoint)[0][len("/run"):])
            self.episode_idx = int(re.findall(r"/episode\d+", model_checkpoint)[0][len("/episode"):])
            self.step_idx = int(re.findall(r"_step\d+", model_checkpoint)[0][len("_step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.run_idx = 0
            while os.path.isdir("./logs/{}/run{}".format(self.model_name, self.run_idx)):
                self.run_idx += 1
            self.episode_idx = 0
            self.step_idx = 0
            os.makedirs("./models/{}/run{}".format(self.model_name, self.run_idx))
        self.train_writer = tf.summary.FileWriter("./logs/{}/run{}".format(self.model_name, self.run_idx), self.sess.graph)

        # Initialize θ_old <- θ
        self.update_old_policy()
        
    def save(self):
        model_checkpoint = "./models/{}/run{}/episode{}_step{}.ckpt".format(self.model_name, self.run_idx, self.episode_idx, self.step_idx)
        self.saver.save(self.sess, model_checkpoint)
        print("Model checkpoint saved to {}".format(model_checkpoint))
        
    def train(self, input_states, taken_actions, returns, advantage, learning_rate=1e-4):
        r = self.sess.run([self.summary_merged, self.train_step, self.loss, self.policy_loss, self.value_loss, self.entropy_loss],
                          feed_dict={self.input_states: input_states,
                                     self.taken_actions: taken_actions,
                                     self.returns: returns,
                                     self.advantage: advantage,
                                     self.learning_rate: learning_rate})
        self.train_writer.add_summary(r[0], self.step_idx)
        self.step_idx += 1
        return r[2:]
        
    def predict(self, input_states, use_old_policy=False):
        policy = self.policy_old if use_old_policy else self.policy
        return self.sess.run([policy.sampled_action, policy.value],
                             feed_dict={self.input_states: input_states})

    def write_episode_summary(self, policy_loss, value_loss, entropy_loss, loss, average_reward, learning_rate):
        summary = tf.Summary()
        summary.value.add(tag="episode_policy_loss", simple_value=policy_loss)
        summary.value.add(tag="episode_value_loss", simple_value=value_loss)
        summary.value.add(tag="episode_entropy_loss", simple_value=entropy_loss)
        summary.value.add(tag="episode_loss", simple_value=loss)
        summary.value.add(tag="episode_average_reward", simple_value=average_reward)
        summary.value.add(tag="episode_learning_rate", simple_value=learning_rate)
        self.train_writer.add_summary(summary, self.episode_idx)
        self.episode_idx += 1

    def update_old_policy(self):
        self.sess.run(self.update_op)
