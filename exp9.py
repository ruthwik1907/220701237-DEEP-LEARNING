import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess the MNIST dataset
(X_train, ), (, _) = tf.keras.datasets.mnist.load_data()
# Normalize images to a range of [-1, 1]
X_train = (X_train.astype('float32') - 127.5) / 127.5
# Reshape for the CNN layers (not used in this simple GAN, but good practice)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# 2. Define the Generator model
def build_generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 256)))
    
    # Upsample to 14x14
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((14, 14, 128)))
    
    # Upsample to 28x28
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((28, 28, 64)))

    model.add(Dense(28*28, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# 3. Define the Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 4. Build the GAN (combined model)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator when training the generator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# 5. Compile and train the GAN
def train_gan(epochs=1, batch_size=128):
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train the Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)
        
        # Print progress and generate example images
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            generate_and_save_images(generator, epoch, 25)

def generate_and_save_images(model, epoch, examples=100):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]
    
    fig = plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# Start the training
train_gan(epochs=2000)