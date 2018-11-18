import tensorflow as tf
import numpy as np
import re
import os

def categorical_entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

class CategoricalA2C():
    def __init__(self, num_actions, optimizer, value_scale=0.5, entropy_scale=0.01, model_checkpoint=None):
        tf.reset_default_graph()
        
        # Construct model
        self.input_states = tf.placeholder(shape=(None, 84, 84, 4), dtype=tf.float32, name="input_states")
        self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), activation="relu", padding="valid")(self.input_states)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv1)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(self.pool1)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv2)
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(self.pool2)
        self.shared_features = tf.keras.layers.Flatten()(self.conv3)
        
        # Policy branch π(a_t | s_t; θ)
        self.action_logits = tf.keras.layers.Dense(num_actions, activation=None)(self.shared_features)
        self.action_prob   = tf.keras.layers.Softmax()(self.action_logits)
        
        # Value branch V(s_t; θ)
        self.value  = tf.keras.layers.Dense(1, activation=None)(self.shared_features)
        
        # Create policy gradient train function
        self.actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32, name="actions_placeholder")
        self.returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.values_placeholder  = tf.placeholder(shape=(None,), dtype=tf.float32, name="values_placeholder")
        self.lr_placeholder      = tf.placeholder(shape=(), dtype=tf.float32, name="lr_placeholder")
        
        # Get probabilities of taken actions: log π(a_t | s_t; θ)
        self.neg_log_action = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_logits,
                                                                             labels=self.actions_placeholder)
        
        # Policy Gradient Loss = ∇_θ log π(a_t | s_t; θ)(R_t − V(s_t; θ_v))
        # Negative log likelihood of the taken actions, weighted by the discounted and normalized rewards
        self.policy_loss  = tf.reduce_mean((self.returns_placeholder - self.values_placeholder) * self.neg_log_action)
        
        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.value), self.returns_placeholder))
        
        # Get entropy
        self.entropy_loss = tf.reduce_mean(categorical_entropy(self.action_logits))
        
        # Total loss
        self.loss = self.policy_loss + self.value_loss * value_scale - self.entropy_loss * entropy_scale
        
        # Minimize loss
        self.optimizer = optimizer(learning_rate=self.lr_placeholder, decay=0.99)
        self.train_step = self.optimizer.minimize(self.loss)
        
        # Create session
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(tf.global_variables_initializer())
        
        tf.summary.scalar("policy_loss", self.policy_loss)
        tf.summary.scalar("value_loss", self.value_loss)
        tf.summary.scalar("entropy_loss", self.entropy_loss)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.lr_placeholder)
        self.summary_merged = tf.summary.merge_all()
        
        # Load model checkpoint if provided
        self.saver = tf.train.Saver()
        if model_checkpoint:
            self.run_idx = int(re.findall(r"_run\d+", model_checkpoint)[0][len("_run"):])
            self.step_idx = int(re.findall(r"_step\d+", model_checkpoint)[0][len("_step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.run_idx = 0
            while os.path.isdir("./logs/run{}".format(self.run_idx)):
                self.run_idx += 1
            self.step_idx = 0
        self.train_writer = tf.summary.FileWriter("./logs/run{}".format(self.run_idx), self.sess.graph)
        
    def save(self):
        model_checkpoint = "./models/a2c_run{}_step{}.ckpt".format(self.run_idx, self.step_idx)
        self.saver.save(self.sess, model_checkpoint)
        print("Model checkpoint saved to {}".format(model_checkpoint))
        
    def train(self, input_states, actions, returns, values, learning_rate=1e-4):
        r = self.sess.run([self.summary_merged, self.train_step, self.loss, self.policy_loss, self.value_loss, self.entropy_loss],
                          feed_dict={self.input_states: input_states,
                                     self.actions_placeholder: actions,
                                     self.returns_placeholder: returns,
                                     self.values_placeholder: values,
                                     self.lr_placeholder: learning_rate})
        self.train_writer.add_summary(r[0], self.step_idx)
        self.step_idx += 1
        return r[2:]
        
    def predict(self, input_states):
        return self.sess.run([self.action_prob, self.value], feed_dict={self.input_states: input_states})
    
def gaussian_entropy(std):
    return tf.reduce_sum(0.5 * np.log(2.0 * np.pi * np.e) + tf.log(std), axis=-1)

class GaussianA2C():
    def __init__(self, num_actions, input_shape, optimizer, action_min, action_max, value_scale=0.5, entropy_scale=0.01, model_checkpoint=None, model_name="a2c"):
        tf.reset_default_graph()
        
        # Construct model
        self.input_states = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state")
        self.conv1 = tf.layers.conv2d(self.input_states, filters=16, kernel_size=8, strides=4, activation=tf.nn.relu, padding="valid", name="conv1")
        self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding="valid", name="conv2")
        self.shared_features = tf.layers.flatten(self.conv2, name="flatten")
        
        # Policy branch π(a_t | s_t; θ)
        self.action_mean = tf.layers.dense(self.shared_features, num_actions, activation=tf.nn.tanh, name="action_mean")
        self.action_std  = tf.layers.dense(self.shared_features, num_actions, activation=tf.nn.softplus, name="action_std") # SoftPlus(x) = log(1 + exp(x))
        self.action_mean = action_min + ((self.action_mean + 1) / 2) * (action_max - action_min)

        # Value branch V(s_t; θ)
        self.conv3 = tf.layers.conv2d(self.input_states, filters=16, kernel_size=8, strides=4, activation=tf.nn.relu, padding="valid", name="conv3")
        self.conv4 = tf.layers.conv2d(self.conv3, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding="valid", name="conv4")
        self.value = tf.layers.dense(tf.layers.flatten(self.conv4), 1, activation=None, name="value")
        
        # Create policy gradient train function
        self.actions_placeholder = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="actions_placeholder")
        self.returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.values_placeholder  = tf.placeholder(shape=(None,), dtype=tf.float32, name="values_placeholder")
        self.lr_placeholder      = tf.placeholder(shape=(), dtype=tf.float32, name="lr_placeholder")
        
        # Get probabilities of taken actions: log π(a_t | s_t; θ)
        #self.neg_log_action = 0.5 * tf.reduce_mean(tf.square((self.actions_placeholder - self.action_mean) / self.action_std), axis=-1) \
                            #+ 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.actions_placeholder)[-1]) \
                            #+ tf.reduce_mean(tf.log(self.action_std), axis=-1) # Alternative: tf.distributions.Normal
        
        self.normal_dist = tf.distributions.Normal(self.action_mean, tf.maximum(self.action_std, 1e-6)) # IDEA: Apply smooth average to self.action_mean
        self.log_action = self.normal_dist.log_prob(self.actions_placeholder)
        self.action = tf.squeeze(self.normal_dist.sample(1), axis=0)
        num_envs = tf.shape(self.action)[0]
        action_min = tf.reshape(tf.tile(tf.convert_to_tensor(action_min, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
        action_max = tf.reshape(tf.tile(tf.convert_to_tensor(action_max, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
        self.action = tf.clip_by_value(self.action, action_min, action_max)

        # Policy Gradient Loss = ∇_θ log π(a_t | s_t; θ)(R_t − V(s_t; θ_v))
        # Negative log likelihood of the taken actions, weighted by the discounted and normalized rewards
        self.policy_loss = tf.reduce_mean((self.returns_placeholder - self.values_placeholder) * -tf.reduce_sum(self.log_action, axis=-1))

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.value), self.returns_placeholder)) * value_scale
        
        # Get entropy
        self.entropy_loss = tf.reduce_mean(self.normal_dist.entropy()) * entropy_scale # gaussian_entropy(self.action_std))
        
        # Total loss
        self.loss = self.policy_loss + self.value_loss - self.entropy_loss
        
        # Minimize loss
        self.optimizer = optimizer(learning_rate=self.lr_placeholder, decay=0.99)
        self.train_step = self.optimizer.minimize(self.loss)
        
        # Create session
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(tf.global_variables_initializer())
        
        # Summaries
        self.policy_loss_summary    = tf.placeholder(dtype=tf.float32, name="policy_loss_summary")
        self.value_loss_summary     = tf.placeholder(dtype=tf.float32, name="value_loss_summary")
        self.entropy_loss_summary   = tf.placeholder(dtype=tf.float32, name="entropy_loss_summary")
        self.loss_summary           = tf.placeholder(dtype=tf.float32, name="loss_summary")
        self.average_reward_summary = tf.placeholder(dtype=tf.float32, name="average_reward_summary")
        tf.summary.scalar("policy_loss", self.policy_loss_summary)
        tf.summary.scalar("value_loss", self.value_loss_summary)
        tf.summary.scalar("entropy_loss", self.entropy_loss_summary)
        tf.summary.scalar("loss", self.loss_summary)
        tf.summary.scalar("average_reward", self.average_reward_summary)
        tf.summary.scalar("learning_rate", self.lr_placeholder)
        self.summary_merged = tf.summary.merge_all()
        
        # Load model checkpoint if provided
        self.model_name = model_name
        self.saver = tf.train.Saver()
        if model_checkpoint:
            self.run_idx = int(re.findall(r"/run\d+", model_checkpoint)[0][len("/run"):])
            self.step_idx = int(re.findall(r"/step\d+", model_checkpoint)[0][len("/step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.run_idx = 0
            while os.path.isdir("./logs/{}/run{}".format(self.model_name, self.run_idx)):
                self.run_idx += 1
            self.step_idx = 0
            os.makedirs("./models/{}/run{}".format(self.model_name, self.run_idx))
        self.train_writer = tf.summary.FileWriter("./logs/{}/run{}".format(self.model_name, self.run_idx), self.sess.graph)
        
    def save(self):
        model_checkpoint = "./models/{}/run{}/step{}.ckpt".format(self.model_name, self.run_idx, self.step_idx)
        self.saver.save(self.sess, model_checkpoint)
        print("Model checkpoint saved to {}".format(model_checkpoint))
        
    def train(self, input_states, actions, returns, values, learning_rate=1e-4):
        r = self.sess.run([self.train_step, self.loss, self.policy_loss, self.value_loss, self.entropy_loss],
                          feed_dict={self.input_states: input_states,
                                     self.actions_placeholder: actions,
                                     self.returns_placeholder: returns,
                                     self.values_placeholder: values,
                                     self.lr_placeholder: learning_rate})
        return r[1:]
        
    def predict(self, input_states):
        return self.sess.run([self.action, self.value],
                             feed_dict={self.input_states: input_states})

    def write_summary(self, policy_loss, value_loss, entropy_loss, loss, average_reward, learning_rate):
        r = self.sess.run(self.summary_merged,
                      feed_dict={self.policy_loss_summary: policy_loss,
                                 self.value_loss_summary: value_loss,
                                 self.entropy_loss_summary: entropy_loss,
                                 self.loss_summary: loss,
                                 self.average_reward_summary: average_reward,
                                 self.lr_placeholder: learning_rate})                      
        self.train_writer.add_summary(r, self.step_idx)
        self.step_idx += 1