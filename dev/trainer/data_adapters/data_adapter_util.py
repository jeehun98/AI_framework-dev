def unpack_x_y_sample_weight(data):
    if isinstance(data, list):
        data = tuple(data)
    
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    error_msg = (
        "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
        f"or `(x, y, sample_weight)`, found: {data}"
    )
    raise ValueError(error_msg)

def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    if y is None:
        
        if not isinstance(x, (tuple, list)):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)

def check_data_cardinality(data):
    pass