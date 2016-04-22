/* Simple code to check whether there a working CUDA runtime + driver + GPU device
 * combination present in the system.
 *
 * The expected result of this program is the CUDA runtime and driver API version
 * printed on the command line and a confirmation that a test kernel has been
 * successfully executed on the CUDA GPU.
 *
 * Compile with:    nvcc check-cuda-drv-runtime.cu -o chk
 * Then run:        ./chk
 * Expected outputs:
 * - everything working fine (CUDA 7.5 driver + runtime):
 *   CUDA driver version: 7050
 *   CUDA runtime version: 7050
 *   Test kernel executed successfully!
 *
 * - no device detected:
 *   CUDA driver version: 7050
 *   cudaRuntimeGetVersion failed: no CUDA-capable device is detected
 *
 * - runtime / driver mismatch (driver ver < runtime ver):
 *   CUDA driver version: 7050
 *   cudaRuntimeGetVersion failed: CUDA driver version is insufficient for CUDA runtime version
 *
 * Author: Szilárd Páll (sin.pecado@gmail.com)
 * 
 */

#include <cstdio>

__global__ void test_kernel() {}

static void check_cuda_retval(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(status));
        exit(1);
    }
}

int main()
{
    cudaError_t stat;
    int rt_ver = 0, drv_ver = 0;

    stat = cudaDriverGetVersion(&drv_ver);
    check_cuda_retval(stat, "cudaDriverGetVersion failed");
    printf("CUDA driver version: %d\n", drv_ver);

    stat = cudaRuntimeGetVersion(&rt_ver);
    check_cuda_retval(stat, "cudaRuntimeGetVersion failed");
    printf("CUDA runtime version: %d\n", rt_ver);

    test_kernel<<<1, 512, 0>>>();
    stat = cudaThreadSynchronize();
    check_cuda_retval(stat, "test kernel launch failed");
    printf("Test kernel executed successfully!\n");

    return 0;
}
