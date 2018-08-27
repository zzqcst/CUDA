## 编程模型
#### 核函数(kernals)
`CUDA C`语言通过允许程序员定义称为`核函数`的C函数来扩展C语言，这些函数在被调用时由N个不同的CUDA线程并行执行N次，而不是像常规C函数那样只执行一次。

使用`__global__`声明说明符定义内核函数，并使用新的`<<<A,B>>>`执行配置语法指定为给定内核调用执行该内核的CUDA线程数,A表示块数，B表示块内线程数。执行核函数的每个线程都有一个唯一的线程ID，可以通过内置的`threadIdx`变量在核函数中访问。

示例，下面的示例代码将两个大小为N的向量A和B相加，并将结果存储到向量C中：
```
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

#### 线程层次结构
`threadIdx`是一个三维的向量，因此可以使用一维、二维、三维线程索引来识别线程。形成了一维、二维、三维的线程块。

线程的索引和它的线程ID以直接的方式相互关联：一维的线程块，二者相同；大小为（Dx,Dy）的二维线程块，线程索引为（x,y）的线程ID是（x+yDx);大小为（Dx,Dy,Dz）的三维线程块，线程索引为（x,y,z)的线程ID是(x + y Dx + z Dx Dy).

![image](https://note.youdao.com/yws/public/resource/a316463a0e3ab0424a29e5a0c00b2395/xmlnote/C5F48D8793A94D57B0D8C2473E98BDF9/2883)

作为示例，以下代码相加两个大小为NxN的矩阵A和B，并将结果存储到矩阵C中：
```
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
每个块的线程数有限制，因为一个线程块块的所有线程都驻留在同一处理器核心上，并且必须共享该核心的有限内存资源。 在当前的GPU上，线程块最多可包含1024个线程。 但是，内核可以由多个同样大小的线程块执行，因此线程总数等于每个块的线程数乘以块数。

线程块组织成一维、二维、或三维线程块网格。
每个块的线程数和`<<<...>>>`语法中指定的每个网格的块数可以是int或dim3类型。可以如上例中那样指定二维块或网格。

网格中的每个块可以通过内核中通过内置的`blockIdx`变量访问的一维，二维或三维索引来识别。 线程块的维度可以通过内置的`blockDim`变量在内核中访问。

多个线程块的示例：
```
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

>blockIdx.x是代表x方向是第几个线程块，从0开始，blockDim.x表示x方向上，一个线程块的宽度，（即x方向上线程块的线程数量）,如下图所示
![image](https://note.youdao.com/yws/public/resource/a316463a0e3ab0424a29e5a0c00b2395/xmlnote/8B449BC128D34446A97579940E1D5804/2919)

线程块需要独立执行：必须能够以任何顺序，并行或串行执行它们。 这种独立性要求允许线程块以任意顺序在任意数量的内核上进行调度，如图5所示，使程序员能够编写随内核数量扩展的代码。

块内的线程可以通过共享存储器共享数据并通过同步它们的执行来协调存储器访问来协作。 更准确地说，可以通过调用`__syncthreads（）`内部函数来指定内核中的同步点; `__syncthreads（）`充当一个屏障，在该屏障中，块中的所有线程必须等待才能允许任何线程继续。 [共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)提供了使用共享内存的示例。 除`__syncthreads（）`之外，[Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)还提供了一组丰富的线程同步原语。

为了实现高效协作，共享内存应该是每个处理器内核附近的低延迟内存（很像L1缓存），而`__syncthreads（）`应该是轻量级的。

#### 分级存储器体系
CUDA线程可以在执行期间从多个内存空间访问数据，如图所示。每个线程都有私有本地内存。 每个线程块都具有对块的所有线程可见的共享内存，并且具有与块相同的生存期。 所有线程都可以访问相同的全局内存。

![image](https://note.youdao.com/yws/public/resource/a316463a0e3ab0424a29e5a0c00b2395/xmlnote/3F796910F8BC43E0B152072CB27D69E8/2942)

所有线程都可以访问两个额外的只读存储空间：常量和纹理存储空间。 全局，常量和纹理内存空间针对不同的内存使用进行了优化（[请参阅设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。 纹理存储器还为某些特定数据格式提供不同的寻址模式以及数据滤波（[请参阅纹理和表面存储器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)）。