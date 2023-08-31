import tensorflow as tf
from tensorflow import keras
from models.generator import Generator
from models.discriminator import Discriminator
import time
from IPython import display
from utils.image_processing import generate_image
from custom_layers.AddGaussianNoise import AddGaussianNoise
import datetime

class Trainer:

    def __init__(self, config):
        self.config = config
        print("Initializing generator and discriminator")
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.gen_optimizer = keras.optimizers.Adam(learning_rate = config['learning_rate'], beta_1 = config['beta_1'], beta_2 = config['beta_2'])
        self.disc_optimizer =  keras.optimizers.Adam(learning_rate = config['learning_rate'], beta_1 = config['beta_1'], beta_2 = config['beta_2'])
        self.summary_writer = tf.summary.create_file_writer(config['log_dir'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer, discriminator_optimizer=self.disc_optimizer, generator=self.generator, discriminator=self.discriminator)
        self.manager = tf.train.CheckpointManager(self.checkpoint, config['checkpoint_dir'], max_to_keep=3)

    def restore_checkpoint(self):
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint:
            print(f"Restoring from {latest_checkpoint}...")
            self.checkpoint.restore(latest_checkpoint)
            return True
        else:
            print("Initializing from scratch.")
            return False
    
    def save_checkpoint(self):
        self.manager.save()

    
    def train_step(self, input_image, target, step, epoch, train_set_len):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            gen_output = self.generator(input_image, training=True)

            # Add Gaussian noise to discriminator inputs
            noise_stddev_L = 0.05  # Standard deviation for L channel
            noise_stddev_AB = 0.1  # Standard deviation for AB channels

            noisy_input_image = input_image + tf.random.normal(shape=tf.shape(input_image), mean=0., stddev=noise_stddev_L)
            noisy_target = target + tf.random.normal(shape=tf.shape(target), mean=0., stddev=noise_stddev_AB)

            # Clip the noisy inputs to be within the valid range
            noisy_input_image = tf.clip_by_value(noisy_input_image, 0, 1)
            noisy_target = tf.clip_by_value(noisy_target, -1, 1)


            noisy_input_image = tf.expand_dims(noisy_input_image, axis = -1)
            disc_real_output = self.discriminator([noisy_input_image, noisy_target], training=True)
            disc_generated_output = self.discriminator([noisy_input_image, gen_output], training=True)

            # Compute the WGAN loss for the discriminator
            disc_real_loss = -tf.reduce_mean(disc_real_output)
            disc_fake_loss = tf.reduce_mean(disc_generated_output)
            disc_loss = disc_real_loss + disc_fake_loss

            # Compute the gradient penalty
            gradient_penalty = self.compute_gradient_penalty(self.discriminator, noisy_target, gen_output, noisy_input_image)
            disc_loss += self.config['LAMBDA_GP'] * gradient_penalty  # typically gradient_penalty_weight = 10

            # WGAN generator loss
            l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
            gen_loss = -tf.reduce_mean(disc_generated_output) + l1_loss * self.config['LAMBDA']



        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        for i in range(self.config['discriminator_ratio']):
            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        writer_step = step + train_set_len * epoch
        with self.summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=writer_step)
            tf.summary.scalar('disc_real_loss', disc_real_loss, step=writer_step)
            tf.summary.scalar('disc_fake_loss', disc_fake_loss, step=writer_step)
            tf.summary.scalar('disc_loss', disc_loss, step=writer_step)
            # If you wish to monitor the gradient penalty separately
            tf.summary.scalar('gradient_penalty', gradient_penalty, step=writer_step)
            tf.summary.scalar('l1_loss', l1_loss, step = writer_step)
    
    
    def fit(self, train_ds, test_ds, epochs):
        train_ds = train_ds.batch(self.config['batch_size'])
        test_ds = test_ds.batch(self.config['batch_size'])   
        train_end = len(train_ds)
        start = time.time()
        DISPLAY_N = 2  # Number of images to display from test set

        print("Training...")
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            train_ds_iter = iter(train_ds)
            print(train_end)

            for step in range(train_end):

                input, target = next(train_ds_iter)

                if (step) == 0:
                    display.clear_output(wait=False)
                    print(f'Time taken for epoch {epoch}:  {time.time() - start} sec\n')
                    start = time.time()
                    test_ds_iter = iter(test_ds.shuffle(buffer_size=len(test_ds)))

                    for n in range(DISPLAY_N // self.config['batch_size']):
                        example_input, example_target = next(test_ds_iter)
                        for i in range(self.config['batch_size']):
                            generate_image(self.generator, example_input[i], example_target[i])



                noisy_input = AddGaussianNoise(stddev = 0.005)(target[..., 0])

                noisy_input = tf.clip_by_value(noisy_input, clip_value_min = 0., clip_value_max = 1.)

                self.train_step(noisy_input, target[..., 1:], step, epoch, train_end)

                if (step + 1) % 1 == 0:
                    print('.')
                    print(f'Epoch {epoch}: {step} / {train_end}')

                if step == train_end - 1:
                    self.manager.save()


    
    def compute_gradient_penalty(self, D, real_samples, fake_samples, real_input):
        alpha = tf.random.normal([real_samples.shape[0], 1, 1, 1], 0.0, 1.0)
        interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as t:
            t.watch(interpolated_samples)
            validity = D([real_input, interpolated_samples])

        gradients = t.gradient(validity, interpolated_samples)
        gradient_penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)

        return gradient_penalty

