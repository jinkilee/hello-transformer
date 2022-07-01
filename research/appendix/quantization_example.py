# Reference: https://leimao.github.io/article/Neural-Networks-Quantization/

import numpy as np

def value_clipping(x, x_max, x_min):
    x[x > x_max] = x_max
    x[x < x_min] = x_min
    return x


def quantization_nbit(mat, s, z, n=8):
    mat = np.round((1/s * mat) + z, decimals=0)
    clipped_mat = value_clipping(mat, 2 ** (n-1) - 1, -2 ** (n-1))
    return clipped_mat.astype(np.int8)


def dequantization(mat_q, s, z):
    mat = s * (mat_q - z)
    mat = mat.astype(np.float32)
    return mat


def make_nbit_quantization_variables(x_min, x_max, n=8):
    xq_min = -2 ** (n - 1)
    xq_max = 2 ** (n - 1) - 1
    s = (x_max - x_min) / (xq_max - xq_min)
    z = int((x_max * xq_min - x_min * xq_max) / (x_max - x_min))
    return s, z


def quantized_matrix_multiplication(x_q, w_q, b_q, s_x, z_x, s_w, z_w, s_b, z_b, s_y, z_y):
    p = w_q.shape[0]
    y_q = z_y + \
            (s_b / s_y * (b_q.astype(np.int32) - z_b)).astype(np.int8) + \
            ((s_x * s_w / s_y) * (np.matmul(x_q.astype(np.int32), w_q.astype(np.int32)) - \
                z_w * np.sum(x_q.astype(np.int32), axis=1, keepdims=True) - \
                z_x * np.sum(w_q.astype(np.int32), axis=0, keepdims=True) + \
                p* z_x * z_w)).astype(np.int8)
    y_q = y_q.astype(np.int8)
    return y_q


def main():
    np.random.seed(0)

    # float32 matrix, x
    min_x_float = -100.0
    max_x_float = 80.0
    s_x, z_x = make_nbit_quantization_variables(min_x_float, max_x_float, n=8)
    x = np.random.uniform(low=min_x_float, high=max_x_float,
                          size=(2, 3)).astype(np.float32)
    x_q = quantization_nbit(x, s_x, z_x, n=8)
    x_dq = dequantization(x_q, s_x, z_x)
    print(f'x = {x}')
    print(f'-> s_x, z_x: {s_x} {z_x}')
    print(f'x_dq = {x_dq}')
    print('---------')

    # float32 matrix, w
    min_w_float = -20.0
    max_w_float = 10.0
    s_w, z_w = make_nbit_quantization_variables(min_w_float, max_w_float, n=8)
    w = np.random.uniform(low=min_w_float, high=max_w_float,
                          size=(3, 4)).astype(np.float32)
    w_q = quantization_nbit(w, s_w, z_w, n=8)
    w_dq = dequantization(w_q, s_w, z_w)
    print(f'w = {w}')
    print(f'-> s_w, z_w: {s_w} {z_w}')
    print(f'w_dq = {w_dq}')
    print('---------')

    # float32 matrix, b
    min_b_float = -500.0
    max_b_float = 500.0
    s_b, z_b = make_nbit_quantization_variables(min_b_float, max_b_float, n=8)
    b = np.random.uniform(low=min_b_float, high=max_b_float,
                          size=(1, 4)).astype(np.float32)
    b_q = quantization_nbit(b, s_b, z_b, n=8)
    b_dq = dequantization(b_q, s_b, z_b)
    print(f'b = {b}')
    print(f'-> s_b, z_b: {s_b} {z_b}')
    print(f'b_dq = {b_dq}')
    print('---------')

    # matrix multiplication with quantized matrix
    min_y_float = -3000.0
    max_y_float = 3000.0
    s_y, z_y = make_nbit_quantization_variables(min_y_float, max_y_float, n=8)
    y_q = quantized_matrix_multiplication(x_q, w_q, b_q, s_x, z_x, s_w, z_w, s_b, z_b, s_y, z_y)
    y_dq = dequantization(y_q, s_y, z_y)
    y = x @ w + b
    print(f'y = {y}')
    print(f'-> s_y, z_y: {s_y} {z_y}')
    print(f'y_dq = {y_dq}')

if __name__ == "__main__":

    main()
