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
        self.input_states = tf.placeholder(shape=(None, 84, 84, 4), dtype=tf.float32)
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
        self.actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.values_placeholder  = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.lr_placeholder      = tf.placeholder(shape=(), dtype=tf.float32)
        
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
    
def gaussian_entropy(variance):
    return tf.reduce_sum(0.5 * tf.log(2.0 * np.pi * np.e * variance), axis=-1)

class GaussianA2C():
    def __init__(self, num_actions, input_shape, optimizer, value_scale=0.5, entropy_scale=0.01, model_checkpoint=None, model_name="a2c"):
        tf.reset_default_graph()
        
        # Construct model
        self.input_states = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32)
        self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), activation="relu", padding="valid")(self.input_states)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv1)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(self.pool1)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv2)
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(self.pool2)
        self.shared_features = tf.keras.layers.Flatten()(self.conv3)
        
        # Policy branch π(a_t | s_t; θ)
        self.action_mean     = tf.keras.layers.Dense(num_actions, activation=None)(self.shared_features)
        self.action_variance = tf.log(1 + tf.exp(tf.keras.layers.Dense(num_actions, activation=None)(self.shared_features))) # SoftPlus(x) = log(1 + exp(x))
        
        # Value branch V(s_t; θ)
        self.value  = tf.keras.layers.Dense(1, activation=None)(self.shared_features)
        
        # Create policy gradient train function
        self.actions_placeholder = tf.placeholder(shape=(None, num_actions), dtype=tf.float32)
        self.returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.values_placeholder  = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.lr_placeholder      = tf.placeholder(shape=(), dtype=tf.float32)
        
        # Get probabilities of taken actions: log π(a_t | s_t; θ)
        self.neg_log_action = 0.5 * tf.reduce_sum(tf.square((self.actions_placeholder - self.action_mean)) / self.action_variance, axis=-1) \
                         + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.actions_placeholder)[-1]) \
                         + tf.reduce_sum(self.action_variance, axis=-1)
        
        # Policy Gradient Loss = ∇_θ log π(a_t | s_t; θ)(R_t − V(s_t; θ_v))
        # Negative log likelihood of the taken actions, weighted by the discounted and normalized rewards
        self.policy_loss = tf.reduce_mean((self.returns_placeholder - self.values_placeholder) * self.neg_log_action)

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.value), self.returns_placeholder))
        
        # Get entropy
        self.entropy_loss = tf.reduce_mean(gaussian_entropy(self.action_variance))
        
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
        self.model_name = model_name
        self.saver = tf.train.Saver()
        if model_checkpoint:
            self.run_idx = int(re.findall(r"_run\d+", model_checkpoint)[0][len("_run"):])
            self.step_idx = int(re.findall(r"_step\d+", model_checkpoint)[0][len("_step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.run_idx = 0
            while os.path.isdir("./logs/{}/run{}".format(self.model_name, self.run_idx)):
                self.run_idx += 1
            self.step_idx = 0
        self.train_writer = tf.summary.FileWriter("./logs/{}/run{}".format(self.model_name,self.run_idx), self.sess.graph)
        
    def save(self):
        model_checkpoint = "./models/{}/a2c_run{}_step{}.ckpt".format(self.model_name,self.run_idx, self.step_idx)
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
        return self.sess.run([self.action_mean, self.action_variance, self.value],
                             feed_dict={self.input_states: input_states})
    