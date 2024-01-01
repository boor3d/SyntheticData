import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from .generator import build_generator
from .discriminator import build_discriminator
from .gan import build_gan
import numpy as np
import matplotlib.pyplot as plt

# Loss function
cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

# Function to generate images
def generate_images(generator, noise_dim, num_images=16):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)
    return generated_images

# Function to display images
def display_images(images, num_rows=4, num_cols=4, scale=1):
    _, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        img = (img + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Training loop
def train_step(generator, discriminator, images, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs, batch_size, noise_dim):
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    generator_losses = []
    discriminator_losses = []

    for epoch in range(epochs):
        epoch_gen_loss = []
        epoch_disc_loss = []
        for image_batch in dataset:
            if isinstance(image_batch, tuple) and len(image_batch) == 2:
                images, _ = image_batch
            else:
                images = image_batch

            gen_loss, disc_loss = train_step(generator, discriminator, images, batch_size, noise_dim)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)

        gen_avg_loss = np.mean(epoch_gen_loss)
        disc_avg_loss = np.mean(epoch_disc_loss)

        generator_losses.append(gen_avg_loss)
        discriminator_losses.append(disc_avg_loss)

        print(f'Epoch {epoch + 1}, Generator Loss: {gen_avg_loss}, Discriminator Loss: {disc_avg_loss}')

    # Generate and display images at the end of training
    generated_images = generate_images(generator, noise_dim, num_images=16)
    display_images(generated_images)

    generator.save('./generative_models/generative_model_storage/generator_model.h5')
    discriminator.save('./generative_models/generative_model_storage/discriminator_model.h5')

    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return generator, discriminator

