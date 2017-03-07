import matplotlib.pyplot as plt
import numpy as np

# Train accuracy plots for each model

basic_train_acc = np.genfromtxt('basic_plain/training_acc.csv')
basic_aug_train_acc = np.genfromtxt('basic_aug_only/training_acc.csv')
basic_vae_train_acc = np.genfromtxt('basic_vae/training_acc.csv')
improved_train_acc = np.genfromtxt('improved_no_vae/training_acc.csv')

plt.figure()
# Batch size for basic - 3000, for vae - 18000
plt.plot(np.array(range(len(basic_train_acc))) * + 1, basic_train_acc, label='basic')
plt.plot(np.array(range(len(basic_aug_train_acc))) + 1, basic_aug_train_acc, label='basic+aug')
plt.plot(np.array(range(len(basic_vae_train_acc))) + 1, basic_vae_train_acc, label='basic+aug+vae')
plt.plot(np.array(range(len(improved_train_acc))) + 1, improved_train_acc, label='improved+aug')
plt.title('Training set accuracy')
plt.ylim(0.8, 1.0)
plt.xlim(0, 100)
plt.xlabel('epoch')
plt.legend()
plt.savefig('plots/training_acc.png')

basic_validation_acc = np.genfromtxt('basic_plain/validation_acc.csv')
basic_aug_validation_acc = np.genfromtxt('basic_aug_only/validation_acc.csv')
basic_vae_validation_acc = np.genfromtxt('basic_vae/validation_acc.csv')
improved_validation_acc = np.genfromtxt('improved_no_vae/validation_acc.csv')

plt.figure()
plt.plot(np.array(range(len(basic_validation_acc))) + 1, basic_validation_acc, label='basic')
plt.plot(np.array(range(len(basic_aug_validation_acc))) + 1, basic_aug_validation_acc, label='basic+aug')
plt.plot(np.array(range(len(basic_vae_validation_acc))) + 1, basic_vae_validation_acc, label='basic+aug+vae')
plt.plot(np.array(range(len(improved_validation_acc))) + 1, improved_validation_acc, label='improved+aug')
plt.ylim(0.8, 1.0)
plt.title('Validation set accuracy')
plt.xlabel('epoch')
plt.xlim(0, 100)
plt.legend()
plt.savefig('plots/validation_acc.png')

plt.figure()
# Combined basic, improved, improved+vae
plt.plot(np.array(range(len(basic_train_acc))) * + 1, basic_train_acc, label='train/basic')
plt.plot(np.array(range(len(improved_train_acc))) + 1, improved_train_acc, label='train/improved+aug')
plt.plot(np.array(range(len(basic_validation_acc))) + 1, basic_validation_acc, label='valid/basic')
plt.plot(np.array(range(len(improved_validation_acc))) + 1, improved_validation_acc, label='valid/improved+aug')
plt.title('Training set accuracy')
plt.ylim(0.8, 1.0)
plt.xlim(0, 100)
plt.xlabel('epoch')
plt.legend()
plt.savefig('plots/combined_acc.png')

bce_loss = np.genfromtxt('bce_training_loss_num.txt')
plt.figure()
plt.plot(np.array(range(len(bce_loss))) * 320, bce_loss, label='bce')
plt.title('VAE model objective for cross-entropy reconstruction loss')
plt.xlabel('num. of examples')
plt.savefig('plots/vae_bce_loss.png')

squared_err_loss = np.genfromtxt('squared_err_training_loss_num.txt')
plt.figure()
plt.plot(np.array(range(len(squared_err_loss))) * 320, squared_err_loss, label='squared_err')
plt.title('VAE model objective for squared error reconstruction loss')
plt.xlabel('num. of examples')
plt.savefig('plots/vae_squared_err_loss.png')
