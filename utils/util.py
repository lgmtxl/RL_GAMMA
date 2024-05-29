def scale_to_0_1(values, min_val=-20, max_val=20):
    return (values - min_val) / (max_val - min_val)
def scale_value_to_minus1_to_1(value, min_val=-20, max_val=20):
    return 2 * ((value - min_val) / (max_val - min_val)) - 1

if __name__ == '__main__':
    values = [-20, -10, 0, 10, 20]
    scaled_values = [scale_value_to_minus1_to_1(v) for v in values]
    print(scaled_values)