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
    """
    Trainer class for training a Wasserstein Generative Adversarial Network (WGAN) for image colorization.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters and settings.
        generator (Generator): The generator model.
        discriminator (Discriminator): The discriminator model.
        gen_optimizer (tf.keras.optimizers.Adam): Optimizer for the generator.
        disc_optimizer (tf.keras.optimizers.Adam): Optimizer for the discriminator.
        summary_writer (tf.summary.SummaryWriter): TensorBoard summary writer.
        checkpoint (tf.train.Checkpoint): Checkpoint object for saving & restoring model states.
        manager (tf.train.CheckpointManager): Manager for handling checkpoints.
    """

    def __init__(self, config):
        """
        Initialize the Trainer class.

        Args:
            config (dict): Configuration dictionary containing hyperparameters and settings.
        """
        self.config = config

        # Initialize generator, discriminator, and their optimizers
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
        )

        # Initialize writer for TensorBoard
        self.summary_writer = tf.summary.create_file_writer(
            config["log_dir"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        # Initialize checkpoints
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, config["checkpoint_dir"], max_to_keep=3
        )

    def restore_checkpoint(self):
        """
        Restore the latest checkpoint if available.

        Returns:
            bool: True if a checkpoint was restored, False otherwise.
        """
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint:
            print(f"Restoring from {latest_checkpoint}...")
            self.checkpoint.restore(latest_checkpoint)
            return True
        else:
            print("Initializing from scratch.")
            return False

    def save_checkpoint(self):
        """
        Save the current checkpoint.
        """
        self.manager.save()

    def fit(self, train_ds, test_ds, epochs):
        """
        Fit the model to the data.

        Args:
            train_ds (tf.data.Dataset): The training dataset.
            test_ds (tf.data.Dataset): The test dataset.
            epochs (int): The number of epochs to train for.
        """

        # Initialize the training set into batches and store its length
        train_ds = train_ds.batch(self.config["batch_size"])
        train_end = len(train_ds)

        # Loop for desired number of epochs
        for epoch in range(epochs):
            # Initialize shuffled iterators for each epoch
            train_ds_iter = iter(train_ds.shuffle(buffer_size=len(train_ds)))
            test_ds_iter = iter(test_ds.shuffle(buffer_size=len(test_ds)))

            # Display examples of images at the beginning of each epoch
            self.display_examples(test_ds_iter)

            # For each step, take new batch of inputs and targets from training set, and take step
            for step in range(train_end):
                input, target = next(train_ds_iter)

                self.train_step(input, target[..., 1:], step, epoch, train_end)

                print(f"Epoch {epoch}: {step} / {train_end}")

            # Save checkpoint at end of each epoch
            self.manager.save()

    def train_step(self, input_image, target, step, epoch, train_set_len):
        """
        Perform a single training step.

        Args:
            input_image (tf.Tensor): Input grayscale images.
            target (tf.Tensor): Target color images.
            step (int): Current step number within the epoch.
            epoch (int): Current epoch number.
            train_set_len (int): Total number of steps in one epoch.
        """
        # Use persistent GradientTape so that multiple steps can be applied to the discriminator at once
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(
            persistent=True
        ) as disc_tape:
            gen_output = self.generator(input_image, training=True)

            # Add noise to the inputs
            noisy_input_image = AddGaussianNoise(0.05)(input_image)
            noisy_target = AddGaussianNoise(0.1)(target)

            # Clip the noisy inputs to be within the valid range
            noisy_input_image = tf.clip_by_value(noisy_input_image, 0, 1)
            noisy_target = tf.clip_by_value(noisy_target, -1, 1)

            # Ensure L channel data has proper shape
            noisy_input_image = tf.expand_dims(noisy_input_image, axis=-1)

            # Input images data into discriminator and store outputs
            disc_real_output = self.discriminator(
                [noisy_input_image, noisy_target], training=True
            )
            disc_generated_output = self.discriminator(
                [noisy_input_image, gen_output], training=True
            )

            # Compute the WGAN loss for the discriminator
            disc_real_loss = -tf.reduce_mean(disc_real_output)
            disc_fake_loss = tf.reduce_mean(disc_generated_output)
            disc_loss = disc_real_loss + disc_fake_loss

            # Compute and add gradient penalty
            gradient_penalty = self.compute_gradient_penalty(
                self.discriminator, noisy_target, gen_output, noisy_input_image
            )
            disc_loss += self.config["LAMBDA_GP"] * gradient_penalty

            # Compute WGAN generator loss
            l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
            gen_loss = (
                -tf.reduce_mean(disc_generated_output) + l1_loss * self.config["LAMBDA"]
            )

        # Apply gradients to generator, and apply number of steps for discriminator as specified in config (used 5 for training, as suggested in WGAN paper)
        self.apply_gradients(gen_tape, gen_loss, self.generator, self.gen_optimizer)
        for _ in range(self.config["discriminator_ratio"]):
            self.apply_gradients(
                disc_tape, disc_loss, self.discriminator, self.disc_optimizer
            )

        # Log metrics at the end of each step
        self.log_metrics(
            step,
            epoch,
            train_set_len,
            gen_loss,
            disc_real_loss,
            disc_fake_loss,
            disc_loss,
            gradient_penalty,
            l1_loss,
        )

    def apply_gradients(self, tape, loss, model, optimizer):
        """
        Apply gradients to the model's trainable variables.

        Args:
            tape (tf.GradientTape): The gradient tape that recorded operations for which to compute gradients.
            loss (tf.Tensor): The loss value to minimize.
            model (tf.keras.Model): The model whose variables should be updated.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for the update.
        """

        # Apply gradients using tape, loss, model, and optimizer
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def display_examples(self, test_ds_iter):
        """
        Display example images from the test dataset.

        Args:
            test_ds_iter (Iterator): An iterator for the test dataset.
        """
        display.clear_output(wait=False)
        # Display number of example images from test set as specified by DISPLAY_N in config
        for _ in range(self.config["DISPLAY_N"]):
            example_input, example_target = next(test_ds_iter)
            generate_image(self.generator, example_input, example_target)

    def compute_gradient_penalty(self, D, real_samples, fake_samples, real_input):
        """
        Compute the gradient penalty for WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty).

        Args:
            D (Discriminator): The discriminator model.
            real_samples (tf.Tensor): Real samples.
            fake_samples (tf.Tensor): Fake samples generated by the generator.
            real_input (tf.Tensor): Real input samples.

        Returns:
            tf.Tensor: Computed gradient penalty.
        """

        # Generate random tensor alpha for mixing real and fake samples
        alpha = tf.random.normal([real_samples.shape[0], 1, 1, 1], 0.0, 1.0)

        # Create interpolated samples by combining real and fake samples based on alpha
        interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples

        # Use GradientTape to record operations for which to compute gradients
        with tf.GradientTape() as t:
            # Make sure to watch the interpolated samples
            t.watch(interpolated_samples)

            # Compute the discriminator's output for the interpolated samples
            validity = D([real_input, interpolated_samples])

        # Compute the gradients of the output with respect to the interpolated samples
        gradients = t.gradient(validity, interpolated_samples)

        # Compute the gradient penalty based on how much these gradients deviate from 1
        gradient_penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)

        return gradient_penalty

    def log_metrics(
        self,
        step,
        epoch,
        train_set_len,
        gen_loss,
        disc_real_loss,
        disc_fake_loss,
        disc_loss,
        gradient_penalty,
        l1_loss,
    ):
        """
        Log metrics to TensorBoard.

        Args:
            step (int): The current training step.
            epoch (int): The current epoch.
            train_set_len (int): The total number of steps in one epoch.
            gen_loss (tf.Tensor): The generator loss.
            disc_real_loss (tf.Tensor): The discriminator real loss.
            disc_fake_loss (tf.Tensor): The discriminator fake loss.
            disc_loss (tf.Tensor): The total discriminator loss.
            gradient_penalty (tf.Tensor): The gradient penalty.
            l1_loss (tf.Tensor): The L1 loss between the generated images and the real images.
        """
        # Write to correct step according to total steps in each epoch and number of epochs already completed
        writer_step = step + train_set_len * epoch

        # Write information into log file
        with self.summary_writer.as_default():
            tf.summary.scalar("gen_loss", gen_loss, step=writer_step)
            tf.summary.scalar("disc_real_loss", disc_real_loss, step=writer_step)
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, step=writer_step)
            tf.summary.scalar("disc_loss", disc_loss, step=writer_step)
            tf.summary.scalar("gradient_penalty", gradient_penalty, step=writer_step)
            tf.summary.scalar("l1_loss", l1_loss, step=writer_step)
