#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float3 cross_product(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float dot_product(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__global__ void solid_angles_kernel(
    const float* __restrict__ points,
    const float* __restrict__ triangles,
    float* __restrict__ output,
    int B, int Q, int F
) {
    int b = blockIdx.x;  // Batch index
    int q = threadIdx.x + blockIdx.y * blockDim.x;  // Query index (handles multi-block)

    if (q >= Q) return;  // Ensure we don’t access out-of-bounds memory

    for (int f = 0; f < F; f++) {
        // Compute correct memory index for points
        int point_index = (b * Q + q) * 3;
        float3 p = make_float3(points[point_index], points[point_index + 1], points[point_index + 2]);

        // Compute correct memory index for triangles
        int triangle_index = (b * F + f) * 9;
        float3 v0 = make_float3(triangles[triangle_index], triangles[triangle_index + 1], triangles[triangle_index + 2]);
        float3 v1 = make_float3(triangles[triangle_index + 3], triangles[triangle_index + 4], triangles[triangle_index + 5]);
        float3 v2 = make_float3(triangles[triangle_index + 6], triangles[triangle_index + 7], triangles[triangle_index + 8]);

        // Center the triangles
        float3 r0 = subtract(v0, p);
        float3 r1 = subtract(v1, p);
        float3 r2 = subtract(v2, p);

        // Compute norms
        float norm_r0 = sqrtf(dot_product(r0, r0));
        float norm_r1 = sqrtf(dot_product(r1, r1));
        float norm_r2 = sqrtf(dot_product(r2, r2));

        // Compute cross product
        float3 cross_prod = cross_product(r1, r2);

        // Compute numerator
        float numerator = dot_product(r0, cross_prod);

        // Compute dot products
        float dot01 = dot_product(r0, r1);
        float dot12 = dot_product(r1, r2);
        float dot02 = dot_product(r0, r2);

        // Compute denominator
        float denominator = (
            norm_r0 * norm_r1 * norm_r2 +
            dot01 * norm_r2 +
            dot02 * norm_r1 +
            dot12 * norm_r0
        );

        // Compute solid angle
        float solid_angle = 2.0f * atan2f(numerator, denominator);

        // Store result
        output[(b * Q + q) * F + f] = solid_angle;
        // if (b == 0 && q == 1 && f == 1) {  // Print values for a specific (B, Q, F)
        //     printf("r0: (%.6f, %.6f, %.6f)\n", r0.x, r0.y, r0.z);
        //     printf("r1: (%.6f, %.6f, %.6f)\n", r1.x, r1.y, r1.z);
        //     printf("r2: (%.6f, %.6f, %.6f)\n", r2.x, r2.y, r2.z);
        //     printf("dot01: %.6f, dot12: %.6f, dot02: %.6f\n", dot01, dot12, dot02);
        //     printf("numerator: %.6f, denominator: %.6f\n", numerator, denominator);
        //     printf("solid_angle: %.6f\n", solid_angle);
        //     printf("points: (%.6f, %.6f, %.6f)\n", p.x, p.y, p.z);
        //     printf("v0: (%.6f, %.6f, %.6f)\n", v0.x, v0.y, v0.z);
        //     printf("v1: (%.6f, %.6f, %.6f)\n", v1.x, v1.y, v1.z);
        //     printf("v2: (%.6f, %.6f, %.6f)\n", v2.x, v2.y, v2.z);
        // }
        // if (b == 0 && f == 1 && (q == 0 || q == 1 || q == 2)) {
        //     printf("b=%d, q=%d, f=%d --> points: (%.6f, %.6f, %.6f)\n", b, q, f, p.x, p.y, p.z);
        // }
    }
}

// C++ function that launches the CUDA kernel
torch::Tensor solid_angles_cuda(torch::Tensor points, torch::Tensor triangles, float thresh) {
    // Ensure contiguous memory to avoid garbage values
    points = points.contiguous();  // Ensure contiguous memory
    triangles = triangles.contiguous();  // Ensure contiguous memory
    const int B = points.size(0);  // Batch size
    const int Q = points.size(1);  // Number of query points
    const int F = triangles.size(1); // Number of faces

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor solid_angles = torch::zeros({B, Q, F}, options);

    // Define the number of threads per block (limit: 1024 per block)
    const int threads_per_block = 256;

    // Compute the number of blocks needed for Q
    const int blocks_Q = (Q + threads_per_block - 1) / threads_per_block;

    // Corrected CUDA kernel launch
    dim3 blocks(B, blocks_Q);  // (Batch, multiple blocks for Q)
    dim3 threads(threads_per_block);  // 256 threads per block

    solid_angles_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        triangles.data_ptr<float>(),
        solid_angles.data_ptr<float>(),
        B, Q, F
    );

    cudaDeviceSynchronize();

    return solid_angles;
}
