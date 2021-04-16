import random


def generate_distribution_from_hist(data):
    dist = []
    for number, occurrencies in data:
        dist += [number] * occurrencies
    random.shuffle(dist)
    return dist
