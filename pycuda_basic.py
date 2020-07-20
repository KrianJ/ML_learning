# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/7/20 19:17"
__doc__ = """ """

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
 const int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N)
 {
  return;
 }
 float temp_a = a[i];
 float temp_b = b[i];
 a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
 // a[i] = a[i] + b[i];
}
""")

func = mod.get_function("func")


def test(N):
    # N = 1024 * 1024 * 90  # float: 4M = 1024 * 1024

    print("N = %d" % N)

    N = np.int32(N)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    # copy a to aa
    aa = np.empty_like(a)
    aa[:] = a
    # GPU run
    nTheads = 256
    nBlocks = int((N + nTheads - 1) / nTheads)
    start = timer()
    func(
        drv.InOut(a), drv.In(b), N,
        block=(nTheads, 1, 1), grid=(nBlocks, 1))
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)
    # cpu run
    start = timer()
    aa = (aa * 10 + 2) * ((b + 2) * 10 - 5) * 5
    run_time = timer() - start

    print("cpu run time %f seconds " % run_time)

    # check result
    r = a - aa
    print(min(r), max(r))


def query_device():
    """查询GPU基本信息"""
    drv.init()
    print('CUDA device query (PyCUDA version) \n')
    print(f'Detected {drv.Device.count()} CUDA Capable device(s) \n')
    for i in range(drv.Device.count()):

        gpu_device = drv.Device(i)
        print(f'Device {i}: {gpu_device.name()}')
        compute_capability = float( '%d.%d' % gpu_device.compute_capability() )
        print(f'\t Compute Capability: {compute_capability}')
        print(f'\t Total Memory: {gpu_device.total_memory()//(1024**2)} megabytes')

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes['MULTIPROCESSOR_COUNT']

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128}[compute_capability]

        print(f'\t ({num_mp}) Multiprocessors, ({cuda_cores_per_mp}) CUDA Cores / Multiprocessor: {num_mp*cuda_cores_per_mp} CUDA Cores')

        device_attributes.pop('MULTIPROCESSOR_COUNT')

        for k in device_attributes.keys():
            print(f'\t {k}: {device_attributes[k]}')


def main():
    for n in range(1, 10):
        N = 1024 * 1024 * (n * 10)
        print("------------%d---------------" % n)
        test(N)


if __name__ == '__main__':
    # main()
    query_device()