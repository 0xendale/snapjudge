from ratelib import rate


def throttle(requests):
    return rate(requests, per_second=5)
