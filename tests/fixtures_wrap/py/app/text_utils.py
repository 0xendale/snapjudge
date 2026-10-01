def rate(count, seconds):
    """Requests per second; unrelated to the review rating wrapper."""
    return count / seconds


def triage(items):
    return sorted(items, key=len)


print(rate(10, 2), triage(["b", "aa"]))
