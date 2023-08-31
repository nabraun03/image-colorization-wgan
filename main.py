from Trainer import Trainer
from utils.data_loader import load_cifar

# Configuration dictionary containing hyperparameters and settings
config = {
    "IMAGE_SIZE": 32,  # Size of the input image
    "LAMBDA": 100,  # Weight for L1 loss in generator
    "LAMBDA_GP": 25,  # Weight for gradient penalty in discriminator
    "batch_size": 16,  # Batch size for training
    "discriminator_ratio": 5,  # Number of discriminator updates per generator update
    "learning_rate": 0.0002,  # Learning rate for Adam optimizers used for generator and discriminator
    "beta_1": 0.5,  # Beta 1 parameter for Adam optimizer used for generator and discriminator
    "beta_2": 0.99,  # Beta 2 parameter for Adam optimizers used for generator and discriminator
    "log_dir": "./logs/",  # Directory for TensorBoard logs
    "checkpoint_dir": "./weights/"
    + "lambda100_bs16_strongergen"
    + ".ckpt",  # Directory for saving model checkpoints
    "DISPLAY_N": 5,  # Number of examples to display before each epoch
}

if __name__ == "__main__":
    # Load the CIFAR-10 dataset, and preprocess it
    print("Loading dataset...")
    train_ds, test_ds = load_cifar()

    # Initialize the Trainer class with the given configuration
    print("Initializing trainer...")
    trainer = Trainer(config)

    # Start the training process
    print("Fitting")
    trainer.fit(train_ds, test_ds, epochs=20)
