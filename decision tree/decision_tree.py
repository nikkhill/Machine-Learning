"""
This file implements a decision tree capable of handling any categorical input and output
to predict handwritten letters
@author nbhale
"""


import math
import numpy as np
from helpers import entropy, read_data
import sys


class Node:
    def __init__(self, depth, is_leaf=False, value=None, attr=None, children=[], label_counts=None):
        self.is_leaf = is_leaf
        self.value = value
        self.children = children
        self.depth = depth
        self.attr = attr
        self.value = value
        self.label_counts = label_counts


def decision_tree(data, remaining_features, depth, max_depth, previous_guess=None):
    # print("Dtree:", data, remaining_features, depth)
    # base case
    # guess
    labels = data[:, -1]
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    if labels.size == 0:
        # if no data, return node with previous guess
        return Node(depth + 1, is_leaf=True, value=previous_guess, label_counts=label_counts)
    guess = unique[np.argmax(counts)]
    # if no more ambiguity or no more features remaining
    if data.shape[0] == np.max(counts) or remaining_features == [] or depth == max_depth:
        return Node(depth + 1, is_leaf=True, value=guess, label_counts=label_counts)

    # Using data, find attr with best info gain (or least conditional
    # probability)
    best_attr = None
    lowest_cond_entropy = None
    for attr in remaining_features:
        # calculate cond entropy
        c_entropy = conditional_entropy(data, attr)
        # check if we found a lower
        if lowest_cond_entropy is None or c_entropy < lowest_cond_entropy:
            best_attr = attr
            lowest_cond_entropy = c_entropy
    # slice data and recurse
    attr_values = data[:, best_attr]
    # unique_attr_values = np.unique(attr_values)
    children = []
    for attr_val in range(len(value_index_dict[best_attr])):  # TODO change this
        # here we assume unique attr values are 0,1,...k (since we converted in
        # data_tointegers)
        data_slice = data[attr_values == attr_val]
        new_remaining_features = remaining_features[:]
        new_remaining_features.remove(best_attr)
        children.append(decision_tree(data_slice, new_remaining_features, depth + 1, max_depth, guess))
    return Node(depth + 1, attr=best_attr, children=children, label_counts=label_counts)


def decision_tree_test(x, node):
    """
    Compute label for unseen data x (a vector or list of attr values)
    """
    if node.is_leaf:
        return node.value
    else:
        attr_xval = x[node.attr]
        return decision_tree_test(x, node.children[attr_xval])


def conditional_entropy(data, attr):
    """
    Calculate conditional entropy H(Y|A), where A is attr (int index)
    """
    attr_values = data[:, attr]
    unique, counts = np.unique(attr_values, return_counts=True)
    attr_value_probabilities = counts / np.sum(counts)
    entropies = []
    for attr_val in unique:
        labels_given_attr_value = data[:, -1][attr_values == attr_val]
        entropies.append(calc_entropy_for_labels(labels_given_attr_value))
    return np.sum(attr_value_probabilities * entropies)


def calc_entropy_for_labels(labels):
    """
    given the labels array find H(Y) [It is already conditioned from caller]
    """
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    num_examples = labels.shape[0]
    label_probabilites = {k: v / num_examples for k, v in label_counts.items()}
    return entropy(label_probabilites)


def data_tointegers(data):
    """
    Convert data string values to integers
    """
    int_data = np.zeros((data.shape[0] - 1, data.shape[1]), dtype=np.int32)
    value_index_dict = {}
    column_headers = data[0]
    for col in range(data.shape[1]):
        col_vec = data[1:, col]
        unique_labels, col_vec_int = np.unique(col_vec, return_inverse=True)
        value_index_dict[col] = unique_labels
        int_data[:, col] = col_vec_int
    return int_data, column_headers, value_index_dict


def print_dtree(node, depth, value_index_dict, col_headers, split_on=""):
    """
    Print the decision tree in given format
    """
    label_val_counts_strings = []
    label_col_index = max(value_index_dict.keys())
    for label_val_index in range(len(value_index_dict[label_col_index])):
        label_val = value_index_dict[label_col_index][label_val_index]
        if label_val_index in node.label_counts:
            count = node.label_counts[label_val_index]
        else:
            count = 0
        label_val_counts_strings.append("{} {}".format(count, label_val.strip()))
    count_string = "[" + " /".join(label_val_counts_strings) + "]"
    print("| " * depth + "{}{}".format(split_on, count_string))
    if not node.is_leaf:
        attr_str = col_headers[node.attr].strip()
        for attr_val in range(len(node.children)):
            split_string = "{} = {}: ".format(attr_str, value_index_dict[node.attr][attr_val])
            print_dtree(node.children[attr_val], depth + 1, value_index_dict, col_headers, split_string)


def compute_output(data, dtree_root):
    X = data[:, :-1]
    Y = np.zeros((X.shape[0],), dtype=np.int32)
    for i in range(X.shape[0]):
        x = X[i, :]
        y = decision_tree_test(x, dtree_root)
        Y[i] = y
    err = np.sum(data[:, -1] != Y) / Y.size
    return Y, err


def write_output(metrics_file, test_err, train_err, test_output_file,
                 test_output, train_output_file, train_output, value_index_dict):
    label_col_index = max(value_index_dict.keys())
    with open(train_output_file, 'w') as trainf:
        for y in train_output:
            trainf.write(value_index_dict[label_col_index][y].strip() + '\n')
    with open(test_output_file, 'w') as testf:
        for y in test_output:
            testf.write(value_index_dict[label_col_index][y].strip() + '\n')
    with open(metrics_file, 'w') as metf:
        metf.write("error(train): {0:.6f}\n".format(train_err))
        metf.write("error(test): {0:.6f}\n".format(test_err))


if __name__ == "__main__":
    _, train_input, test_input, max_depth, train_output_file, test_output_file, metrics_file = sys.argv
    train_data = read_data(train_input)
    train_data, col_headers, value_index_dict = data_tointegers(train_data)
    root = decision_tree(train_data, list(range(train_data.shape[1] - 1)), 0, int(max_depth))
    print_dtree(root, 0, value_index_dict, col_headers)
    train_output, train_err = compute_output(train_data, root)
    test_data = read_data(test_input)
    test_data, _, _ = data_tointegers(test_data)
    test_output, test_err = compute_output(test_data, root)
    print("test_err, train err = ", test_err, train_err)
    write_output(metrics_file, test_err, train_err, test_output_file, test_output,
                 train_output_file, train_output, value_index_dict)
