from app.rating import rate


def summarize(scores):
    def rate(value):
        return round(value, 1)

    return [rate(score) for score in scores]
