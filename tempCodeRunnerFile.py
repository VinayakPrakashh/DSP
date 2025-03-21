    # Append remaining bits if the strings have different lengths
    if len(a) > len(b):
        result.extend(a[min_len:])
    elif len(b) > len(a):
        result.extend(b[min_len:])