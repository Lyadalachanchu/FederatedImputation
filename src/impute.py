import numpy as np
import torch.utils.data

from src.vae.mnist_vae import VaeAutoencoderClassifier


def impute_naive(k, trained_vae:VaeAutoencoderClassifier, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = trained_vae.generate_data(n_samples=k)
    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[0][image_ind], np.argmax(generated_dataset[1][image_ind].detach().numpy())))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])