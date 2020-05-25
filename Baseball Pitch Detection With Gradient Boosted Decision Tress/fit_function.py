# !/usr/bin/python3
# -*- coding: utf-8 -*-
# __author__ = "Grayson_Wu"
import numpy as np
import json
import os
import const
import matplotlib.pyplot as plt


json_filepath = './videos/dataCollection/'


def load_json(filename):

    xs = []
    ys = []

    with open(filename) as json_file:
        print(filename)
        data = json.load(json_file)
        for p in data['balls']:
            xs.append(int(p['X']))
            ys.append(720 - int(p['Y']))

        label = const.LABEL_TO_INT[data["label"]]
        hand = const.HAND_MAP[data["Hand"]]
        frame = int(data["frames"])

    return xs, ys, xs[0], xs[len(xs)-1], label, hand, frame


def plot_check(factors, xs, ys, filename):

    p1 = np.poly1d(factors)
    yvals = p1(xs)

    plt.plot(xs, ys, 's', label='original values')
    plt.plot(xs, yvals, 'r', label='polyfit values')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([200, 1000, 200, 600])
    plt.legend(loc=4)
    plt.title(filename)
    # plt.show()
    plt.savefig('./fit_function/' + filename.split('.')[0] + '.png')
    plt.clf()


def get_fit_func(xs, ys, filename):

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    factors = np.polyfit(xs, ys, 2)
    # factors = np.polynomial.Polynomial.fit(xs, ys, 4)
    # print('factor :\n', factors)

    # plot_check(factors, xs, ys, filename)

    return factors


def main():

    files = os.listdir(json_filepath)

    train_data = []
    train_label = []
    for filename in files:

        xs, ys, start, end, label, hand, frame = load_json(json_filepath + filename)

        params = get_fit_func(xs, ys, filename)

        data = [param for param in params]
        data.append(start)
        data.append(end)
        data.append(hand)
        data.append(frame)

        train_data.append(data)
        train_label.append(label)

    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)

    np.savetxt("train_data.csv", train_data, delimiter=",")
    np.savetxt("train_label.csv", train_label, delimiter=",")


if __name__ == '__main__':
    main()
