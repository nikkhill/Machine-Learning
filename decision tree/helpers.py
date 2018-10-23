import sys
import math
import numpy as np


def entropy_one_label(p):
    if p == 0:
        return 0
    return - (p * math.log(p, 2))


def entropy(label_probabilities):
    e = 0
    for label, p in label_probabilities.items():
        e += entropy_one_label(p)
    return e


def read_data(in_file):
    with open(in_file) as inf:
        data = [line.split(',') for line in inf.readlines()]
    data = np.array(data)
    return data


def process_input(data):
    labels = data[1:, -1]
    # unique_labels, y = np.unique(labels, return_inverse=True)
    unique, counts = np.unique(labels, return_counts=True)
    # print(unique_labels, y)
    label_counts = dict(zip(unique, counts))
    num_examples = labels.shape[0]
    label_probabilites = {k: v / num_examples for k, v in label_counts.items()}
    max_label_index = np.argmax(counts)
    majority_vote_error = (num_examples - counts[max_label_index]) / num_examples
    return entropy(label_probabilites), majority_vote_error


def write_inspect_output(output_file, entropy_val, error_val):
    with open(output_file, 'w') as out_file:
        out_file.write("entropy: {0:.12f}\n".format(entropy_val))
        out_file.write("error: {0:.12f}".format(error_val))


if __name__ == "__main__":
    inf = sys.argv[1]
    outf = sys.argv[2]
    data = read_data(inf)
    en, er = process_input(data)
    write_inspect_output(outf, en, er)
