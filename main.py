from Trainer import Trainer
from utils.data_loader import load_cifar
config = {
    'IMAGE_SIZE': 32,
    'LAMBDA': 100,
    'LAMBDA_GP': 25,
    'batch_size': 16,
    'discriminator_ratio': 5,
    'learning_rate': 0.0002,
    'beta_1': 0.5,
    'beta_2': 0.99,
    'log_dir': './logs/',
    'checkpoint_dir': './weights/' + 'lambda100_bs16_strongergen' + '.ckpt'
}

if __name__ == '__main__':

    print("Loading dataset...")
    train_ds, test_ds = load_cifar()

    print("Initializing trainer...")
    trainer = Trainer(config)
    print("Fitting")
    trainer.fit(train_ds, test_ds, epochs = 20)
