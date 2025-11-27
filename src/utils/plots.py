import numpy as np
import matplotlib.pyplot as plt


def plot_sample_images(dataset, class_names, num_images=9):
    """Visualiza amostras do dataset"""
    plt.figure(figsize=(12, 12))

    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            # Pegar nome da classe
            label_idx = np.argmax(labels[i])
            plt.title(class_names[label_idx])
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plota o histórico de treinamento do modelo"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treino e Validação')

    plt.show()
