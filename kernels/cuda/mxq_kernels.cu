extern "C" __global__ void axonal_dequant_mxq_kernel(
    const unsigned char* packed,
    const half* scales,
    float* output,
    int values_per_group,
    int bits,
    int group_count
) {
    int group_index = blockIdx.x;
    int local_index = threadIdx.x;
    if (group_index >= group_count || local_index >= values_per_group) {
        return;
    }

    int zero_point = 1 << (bits - 1);
    int bit_index = local_index * bits;
    int byte_index = bit_index >> 3;
    int shift = bit_index & 7;
    unsigned int word = packed[group_index * ((values_per_group * bits + 7) / 8) + byte_index];
    if (byte_index + 1 < ((values_per_group * bits + 7) / 8)) {
        word |= ((unsigned int)packed[group_index * ((values_per_group * bits + 7) / 8) + byte_index + 1]) << 8;
    }
    if (byte_index + 2 < ((values_per_group * bits + 7) / 8)) {
        word |= ((unsigned int)packed[group_index * ((values_per_group * bits + 7) / 8) + byte_index + 2]) << 16;
    }
    unsigned int code = (word >> shift) & ((1u << bits) - 1u);
    output[group_index * values_per_group + local_index] = (__half2float(scales[group_index]) * ((int)code - zero_point));
}

extern "C" __global__ void axonal_fused_mxq_matvec_kernel(
    const float* dequantized,
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int col = 0; col < cols; ++col) {
        acc += dequantized[row * cols + col] * input[col];
    }
    output[row] = acc;
}
