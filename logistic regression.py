#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
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


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    dumbcolumn = np.full((np.shape(data)[0], 1), 1)
    data1 = np.append(data, dumbcolumn, axis = 1)
    z = np.dot(data1, weights)
    y = sigmoid(z)
    return y


# In[4]:


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          D is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    LCE_vector = (-targets * np.log(y)) - (1-targets)*np.log(1-y)
    ce = np.average(LCE_vector)
    prediction = np.where(y<=0.5, 0, 1)
    Numcorrect = np.sum(prediction == targets)
    frac_correct = Numcorrect/np.shape(y)[0]
    return ce, frac_correct


# In[5]:


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (D+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of `probabilities`.
    """
    y = logistic_predict(weights, data)
    ce, frac_correct = evaluate(targets, y)
    lambd = hyperparameters["weight_regularization"]
    penalty = (np.sum(np.square(weights))*lambd)/2
    f = ce + penalty
    dumbcolumn = np.full((np.shape(data)[0], 1), 1)
    data1 = np.append(data, dumbcolumn, axis = 1)
    df = np.reshape(np.average((y-targets)*data1, axis = 0),(np.shape(weights)[0],1)) + lambd * weights
    return f, df, y


# In[6]:


def run_logistic_regression():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    # If you would like to use digits_train_small, please uncomment this line:
    # x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    n, d = x_train.shape


    hyperparameters = {
        "learning_rate": 0.02,
        "weight_regularization": 0.
        "num_iterations": 3000
    }
    # train model
    weights = np.zeros((d + 1, 1))
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights,x_train,y_train,hyperparameters)
        weights = weights - df*hyperparameters["learning_rate"]
    ce_train, frac_correct = evaluate(y_train, y)
    train_error = 1 - frac_correct
    #Tuning parameter using valid
    f_val, df_val, y_val = logistic(weights, x_valid, y_valid, hyperparameters)
    # plot_digits(y_val)
    # plot_digit(y_valid)
    ce_val, frac_correct = evaluate(y_valid, y_val)
    val_error = 1 - frac_correct
    #apply test set
    f_tst, df_tst, y_tst = logistic(weights, x_test, y_test, hyperparameters)
    ce_tst, frac_correct = evaluate(y_test, y_tst)
    test_error = 1- frac_correct
    return train_error, val_error, test_error, ce_train,ce_val,ce_tst



# In[7]:

run_logistic_regression()
print(run_logistic_regression())

