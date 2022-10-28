import pickle
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from util import batch_hessian_tf


class NNRiskModel(tf.keras.layers.Layer):

    def __init__(self, state_dim, action_dim, hidden=256, l2=0.0001, lr=0.0001, EPS=10**(-8), use_l2=False, dropout_rate=0.5, use_dropout=True):
        self.state_ph = tf.keras.layers.Input(state_dim, name='state')
        self.action_ph = tf.keras.layers.Input(action_dim, name='action')
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.l1 = None
        if use_l2:
            self.l1 = tf.keras.layers.Dense(hidden, activation='tanh', name='l1', kernel_regularizer=tf.keras.regularizers.l2(l2))
        else:
            self.l1 = tf.keras.layers.Dense(hidden, activation='tanh', name='l1')

        if use_dropout:
            self.b1 = tf.keras.layers.Dropout(dropout_rate)

        self.l2 = None
        if use_l2:
            self.l2 = tf.keras.layers.Dense(hidden, activation='tanh', name='l2', kernel_regularizer=tf.keras.regularizers.l2(l2))
        else:
            self.l2 = tf.keras.layers.Dense(hidden, activation='tanh', name='l2')

        if use_dropout:
            self.b2 = tf.keras.layers.Dropout(dropout_rate)

        self.l3_mu = tf.keras.layers.Dense(1, name='mu')
        self.l3_logstd = tf.keras.layers.Dense(1, name='logstd')

        self.sess = tf.keras.backend.get_session()

        self.state_action = self.concat([self.state_ph, self.action_ph])
        h = self.l1(self.state_action)
        if use_dropout:
            h = self.b1(h)
        h = self.l2(h)
        if use_dropout:
            h = self.b2(h)
        self.risk_mu = self.l3_mu(h)
        self.risk_std = tf.exp(self.l3_logstd(h))
        risk_dist = tfp.distributions.Normal(self.risk_mu, self.risk_std + EPS)
        self.sampled_risk = risk_dist.sample()

        self.mean_hess = batch_hessian_tf(self.risk_mu[:, 0], self.action_ph)
        self.second_moment_r = self.risk_std ** 2 + self.risk_mu ** 2

        def loss(y_true, y_pred):
            log_likelihood = risk_dist.log_prob(y_true)
            return -tf.reduce_mean(log_likelihood)

        self.model = tf.keras.models.Model(inputs=[self.state_ph, self.action_ph], outputs=[self.sampled_risk])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)


    def train(self, states, actions, risks, batch_size=256):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        train_info = self.model.fit(x=[states, actions], y=risks, validation_split=0.2, batch_size=int(batch_size), epochs=10000, callbacks=[early_stopping], verbose=0)
        return train_info

    def get_mean_hess(self, states, actions):
        mean_hess_value = self.sess.run(self.mean_hess, feed_dict={self.state_ph: states, self.action_ph: actions})
        return mean_hess_value

    def get_2nd_moment_of_r(self, states, actions):
        second_moment_r_value = self.sess.run(self.second_moment_r, feed_dict={self.state_ph: states, self.action_ph: actions})
        return second_moment_r_value

    def get_mean_std(self, states, actions):
        mean, std = self.sess.run([self.risk_mu, self.risk_std], feed_dict={self.state_ph: states, self.action_ph: actions})
        return mean, std

    def save(self, filepath):
        self.model.save_weights(filepath = filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)
