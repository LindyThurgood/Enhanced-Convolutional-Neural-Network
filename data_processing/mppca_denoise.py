import numpy as np
import h5py
from scipy.io import loadmat
from numpy.linalg import svd

'''This script implements a sliding-window MP-PCA denoiser for 3D image data. 
It was taken from the original MATLAB code created by NeuroPhysics at CFIN, Aarhus University, and adapted to Python with the help AI. 
Matlab code source link: https://github.com/Neurophysics-CFIN/MP-PCA-Denoising'''

# Helper: imageAssert
def imageAssert(image, mask):
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., None]
    if mask is None:
        mask = np.ones(image.shape[:3], dtype=bool)
    return image, mask



# Placeholder MP-PCA denoiser
def denoiseMatrix(X):
    U, S, Vt = svd(X, full_matrices=False)

    thresh = np.median(S)
    k = np.sum(S > thresh)

    if k == 0:
        return X, 0.0, 0

    Xd = (U[:, :k] * S[:k]) @ Vt[:k]
    sigma2 = np.mean(S[k:]**2) if k < len(S) else 0.0

    return Xd, sigma2, k



# Sliding-window denoise
def denoise(image, window, mask=None):

    dimsOld = image.shape
    image = image.astype(np.float32)
    image, mask = imageAssert(image, mask)
    dims = image.shape

    window = list(window)
    if len(window) == 2:
        window.append(1)

    denoised = np.zeros_like(image)
    count = np.zeros(dims[:3])

    M = dims[0] - window[0] + 1
    N = dims[1] - window[1] + 1
    O = dims[2] - window[2] + 1

    for k in range(O):
        for j in range(N):
            for i in range(M):

                rows = slice(i, i + window[0])
                cols = slice(j, j + window[1])
                slis = slice(k, k + window[2])

                block = image[rows, cols, slis, :]
                X = block.reshape(-1, dims[3]).T

                mask_block = mask[rows, cols, slis].reshape(-1)
                if np.count_nonzero(mask_block) <= 1:
                    continue

                Xd, _, _ = denoiseMatrix(X[:, mask_block])
                X[:, mask_block] = Xd
                X[:, ~mask_block] = 0

                denoised[rows, cols, slis, :] += X.T.reshape(window[0], window[1], window[2], dims[3])
                count[rows, cols, slis] += 1

    skipped = (count == 0) | (~mask)
    denoised += image * skipped[..., None]
    count[skipped] = 1
    denoised /= count[..., None]

    return denoised.reshape(dimsOld)



# Main pipeline
def main():
    print("Loading .mat file...")
    data = loadmat("olivetti_data.mat")

    images = data["images"]
    labels = data["labels"]

    print("Images:", images.shape, images.dtype)
    print("Labels:", labels.shape)

    images = images.astype(np.float32)

    print("Denoising...")  
    denoised = denoise(images, window=(5, 5, 3))

    print("Saving to HDF5...")

    with h5py.File("olivetti_denoised.h5", "w") as f:
        f.create_dataset("images", data=denoised, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")

    print("Done.")


if __name__ == "__main__":
    main()
