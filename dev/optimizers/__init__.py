from dev.optimizers.sgd import SGD


ALL_OPTIMIZERS = {
    SGD,
}

ALL_OPTIMIZERS_DICT = {cls.__name__.lower(): cls for cls in ALL_OPTIMIZERS}

def get(identifier):
    if isinstance(identifier, str):
        obj = ALL_OPTIMIZERS_DICT.get(identifier, None)

    return obj