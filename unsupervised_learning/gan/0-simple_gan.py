#!/usr/bin/env python3
"""Simple GAN class"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Simple generative adversarial network"""
    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2, learning_rate=.005):
        """Initializer"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss)

        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Gets a fake sample"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Gets a real sample"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Trains the discriminator and generator"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get real and fake samples
                real_sample = self.discriminator(self.get_real_sample())
                fake_sample = self.discriminator(self.get_fake_sample())
                # Compute loss
                discr_loss = self.discriminator.loss(real_sample, fake_sample)
            # Compute gradients
            gradients = tape.gradient(discr_loss,
                                      self.discriminator.trainable_variables)
            # Apply gradients
            self.discriminator.optimizer.apply_gradients(
                 zip(gradients, self.discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                # Get fake sample
                size = self.batch_size
                generated = self.generator(self.latent_generator(size))
                fake_sample = self.discriminator(generated)
                # Compute loss
                gen_loss = self.generator.loss(fake_sample)
            # Compute gradients
            gradients = tape.gradient(gen_loss,
                                      self.generator.trainable_variables)
            # Apply gradients
            self.generator.optimizer.apply_gradients(
                zip(gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
