"""
Author:  Kaiwen Gong
Date last modified: 01/26/2024
Organization: CSE 6740 Class

Description:
Implementation of kmeans and kmedoids
modify code from: https://www.geeksforgeeks.org/image-compression-using-k-means-clustering/
"""
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import numpy as np
import imageio
from matplotlib import pyplot as plt
import os
import time



def mykmeans(pixels, K):

    pixels = pixels.reshape((-1, 3)).astype(float)


    m, n = pixels.shape
    indices = np.random.choice(m, K, replace=False)
    means = pixels[indices]

    assignments = np.zeros(m, dtype=int)
    iterations = 10
    while iterations > 0:
        iterations -= 1
        prev_assignments = assignments.copy()

        distances = np.linalg.norm(pixels[:, None, :] - means[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)


        for k in range(K):
            cluster_points = pixels[assignments == k]
            if len(cluster_points) > 0:
                means[k] = np.mean(cluster_points, axis=0)


        if np.array_equal(assignments, prev_assignments):
            break

    recovered = means[assignments].reshape(-1, n)
    recovered = np.clip(recovered, 0, 1)
    recovered_image = (recovered * 255).astype(np.uint8)

    return assignments, means





import numpy as np

def mykmedoids(pixels, K):
    pixels = pixels.reshape((-1, 3)).astype(float)
    m, n = pixels.shape

    indices = np.random.choice(m, K, replace=False)
    medoids = pixels[indices]

    assignments = np.zeros(m, dtype=int)
    iteration = 10
    while iteration > 0:
        iteration -= 1
        prev_assignments = assignments.copy()

        distances = np.sum(np.abs(pixels[:, None, :] - medoids[None, :, :]), axis=2)
        assignments = np.argmin(distances, axis=1)


        for k in range(K):
            cluster_points = pixels[assignments == k]
            if len(cluster_points) > 0:
                mean_point = np.mean(cluster_points, axis=0)
                distances_to_mean = np.sum(np.abs(cluster_points - mean_point), axis=1)
                new_medoid_index = np.argmin(distances_to_mean)
                medoids[k] = cluster_points[new_medoid_index]

        # 检查是否收敛
        if np.array_equal(assignments, prev_assignments):
            break

    return assignments, medoids




def main():
    if (len(sys.argv) < 2):
        print("Please supply an image file")
        return

    image_file_name = sys.argv[1]
    K = 16 if len(sys.argv) == 2 else int(sys.argv[2])
    print(image_file_name, K)
    im = np.asarray(imageio.imread(image_file_name))

    fig, axs = plt.subplots(1, 2)
    T1 = time.time()
    classes, centers = mykmedoids(im, K)
    T2 = time.time()
    print(classes, centers)
    print("Time for mykmedoids:%s"% ((T2 - T1) * 1000 ))
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) +
                    os.path.splitext(image_file_name)[1], new_im)
    axs[0].imshow(new_im)
    axs[0].set_title('K-medoids')
    T3 = time.time()
    classes, centers = mykmeans(im, K)
    T4 = time.time()
    print(classes, centers)
    print("Time for mykmeans:%s" % ((T4 - T3) * 1000))
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) +
                    os.path.splitext(image_file_name)[1], new_im)
    axs[1].imshow(new_im)
    axs[1].set_title('K-means')

    plt.show()


def compress_image(image_file_name, K, method='kmeans'):
    im = np.asarray(imageio.imread(image_file_name))

    if method == 'kmeans':
        T1 = time.time()
        classes, centers = mykmeans(im, K)
        T2 = time.time()
    elif method == 'kmedoids':
        T1 = time.time()
        classes, centers = mykmedoids(im, K)
        T2 = time.time()
    else:
        return

    print(f"Time for {method}:{(T2 - T1) * 1000}ms")
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    output_filename = os.path.basename(os.path.splitext(image_file_name)[0]) + f'_converted_{method}_' + str(K) + \
                      os.path.splitext(image_file_name)[1]
    imageio.imwrite(output_filename, new_im)
    plt.imshow(new_im)
    plt.title(f'{method.capitalize()} with K={K}')
    plt.show()


def choose_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return filename


def get_k_value():
    K = simpledialog.askinteger("Input", "Enter the number of clusters (K)", minvalue=1, maxvalue=256)
    return K


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    image_file_name = choose_file()
    if image_file_name:  # Proceed only if a file was selected
        K = get_k_value()
        if K:  # Proceed only if a value for K was given
            method = simpledialog.askstring("Input", "Enter the method (kmeans or kmedoids):")
            compress_image(image_file_name, K, method)


if __name__ == '__main__':
    main()