#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy.linalg as lin
import numpy as np
import requests
import io


# In[2]:


def sigmoid(x):
    """ Computes the element wise logistic sigmoid of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_train():
    """ Loads training data for digits_train.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f21/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    train_inputs = np.hstack((data["train2"], data["train3"]))
    train_targets = np.hstack((np.zeros((1, data["train2"].shape[1])), np.ones((1, data["train3"].shape[1]))))
    return train_inputs.T, train_targets.T


def load_train_small():
    """ Loads training data for digits_train_small.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f21/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    train_inputs = np.hstack((data["train2"][:, :2], data["train3"][:, :2]))
    train_targets = np.hstack((np.zeros((1, 2)), np.ones((1, 2))))
    return train_inputs.T, train_targets.T


def load_valid():
    """ Loads validation data.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f21/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    valid_inputs = np.hstack((data["valid2"], data["valid3"]))
    valid_targets = np.hstack((np.zeros((1, data["valid2"].shape[1])), np.ones((1, data["valid3"].shape[1]))))
    return valid_inputs.T, valid_targets.T


def load_test():
    """ Loads validation data.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f21/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    test_inputs = np.hstack((data["test2"], data["test3"]))
    test_targets = np.hstack((np.zeros((1, data["test2"].shape[1])), np.ones((1, data["test3"].shape[1]))))
    return test_inputs.T, test_targets.T


def plot_digits(digit_array):
    """ Visualizes each example in digit_array.
    :param digit_array: N x D array of pixel intensities.
    :return: None
    """
    CLASS_EXAMPLES_PER_PANE = 5

    # assume two evenly split classes
    examples_per_class = int(digit_array.shape[0] / 2)
    num_panes = int(np.ceil(float(examples_per_class) / CLASS_EXAMPLES_PER_PANE))

    for pane in range(num_panes):
        print("Displaying pane {}/{}".format(pane + 1, num_panes))

        top_start = pane * CLASS_EXAMPLES_PER_PANE
        top_end = min((pane + 1) * CLASS_EXAMPLES_PER_PANE, examples_per_class)
        top_pane_digits = extract_digits(digit_array, top_start, top_end)

        bottom_start = top_start + examples_per_class
        bottom_end = top_end + examples_per_class
        bottom_pane_digits = extract_digits(digit_array, bottom_start, bottom_end)

        show_pane(top_pane_digits, bottom_pane_digits)


def extract_digits(digit_array, start_index, end_index):
    """ Returns a list of 16 x 16 pixel intensity arrays starting
    at start_index and ending at end_index.
    """
    digits = []
    for index in range(start_index, end_index):
        digits.append(extract_digit_pixels(digit_array, index))
    return digits


def extract_digit_pixels(digit_array, index):
    """ Extracts the 16 x 16 pixel intensity array at the specified index.
    """
    return digit_array[index].reshape(16, 16).T


def show_pane(top_digits, bottom_digits):
    """ Displays two rows of digits on the screen.
    """
    all_digits = top_digits + bottom_digits
    fig, axes = plt.subplots(nrows=2, ncols=int(len(all_digits) / 2))
    for axis, digit in zip(axes.reshape(-1), all_digits):
        axis.imshow(digit, interpolation="nearest", cmap=plt.gray())
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.axis("off")
    # fig.subplots_adjust(wspace=0,
    #                     hspace=0)
    plt.tight_layout(h_pad=-7)
    plt.show()


def save_images(images, filename):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_digits(images)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


# In[3]:


def show_eigenvectors(v):
    """ Display the eigenvectors as images.
    :param v: NumPy array
        The eigenvectors
    :return: None
    """
    plt.figure(1)
    plt.clf()
    for i in range(v.shape[1]):
        plt.subplot(1, v.shape[1], i + 1)
        plt.imshow(v[:, v.shape[1] - i - 1].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.show()


# In[4]:


def pca(x, k):
    """ PCA algorithm. Given the data matrix x and k,
    return the eigenvectors, mean of x, and the projected data (code vectors).

    Hint: You may use NumPy or SciPy to compute the eigenvectors/eigenvalues.

    :param x: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param k: int
        Number of dimension to reduce to.
    :return: Tuple of (Numpy array, Numpy array, Numpy array)
        WHERE
        v: A matrix of dimension D x k that stores top k eigenvectors
        mean: A vector of dimension D x 1 that represents the mean of x.
        proj_x: A matrix of dimension k x N where x is projected down to k dimension.
    """
    n, d = x.shape
    mean = np.reshape(np.average(x, axis = 0),(1,d))
    covariance = np.dot(np.transpose(x - mean),x-mean)/n
    eigenValues, eigenVectors = np.linalg.eig(covariance)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    v = eigenVectors[:,1:k+1]
    proj_x = np.dot(np.transpose(v),np.transpose(x-mean))
    reported_mean = np.reshape(mean,(d,1))
    return v, reported_mean, proj_x


# In[5]:


def pca_classify():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    v, mean, proj_x = pca(x_train, 5)
    # The below code visualize the eigenvectors.
    show_eigenvectors(v)

    k_lst = [2, 5, 10, 20, 30]
    val_acc = np.zeros(len(k_lst))
    #choose appropriate eigenvector and means from the training dataset (basis and datacenter) for validation set to project on
    #this is the same as choosing a good subspace first using training set
    for j, k in enumerate(k_lst):
        v, mean_reported, proj_x = pca(x_train, k)
        train_mean = np.reshape(mean_reported,(1,np.shape(x_train)[1]))
        proj_val = np.dot(np.transpose(v), np.transpose(x_valid - train_mean))
        valid_reconstruction = np.transpose(np.dot(v, proj_val) + np.transpose(train_mean))
        abs_difference_matrix = np.absolute(x_valid-valid_reconstruction)
        desired_value = np.amin(abs_difference_matrix, axis=1)
        desired_index_list = []
        for i in range(x_valid.shape[0]):
            desired_index = abs_difference_matrix[i].tolist().index(desired_value[i])
            desired_index_list.append(desired_index)
        collapsed_column = []
        for i in range(x_valid.shape[0]):
            collapsed_column.append(x_valid[i][desired_index_list[i]])
        collapsed_array = np.reshape(np.array(collapsed_column),(len(collapsed_column),1))
        prediction = np.where(collapsed_array <= 0.5, 0, 1)
        Numcorrect = np.sum(prediction == y_valid)
        frac_correct = Numcorrect / np.shape(y_valid)[0]
        val_acc[j] = frac_correct

    return val_acc
    plt.plot(k_lst, val_acc)
    plt.show()


# In[6]:


pca_classify()
print(pca_classify())


# In[ ]:




