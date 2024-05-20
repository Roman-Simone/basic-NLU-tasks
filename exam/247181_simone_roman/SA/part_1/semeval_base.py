
# Aspect Category Detection
def category_detection(size, correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(size):
        cor = correct[i].get_aspect_categories()
        # Use set to avoid duplicates (i.e., two times the same category)
        pre = set(predicted[i].get_aspect_categories())
        common += len([c for c in pre if c in cor])
        retrieved += len(pre)
        relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (1 + b ** 2) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
    return f1



