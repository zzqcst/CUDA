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

全局，常量和纹理内存空间在同一应用程序的内核启动之间是持久存在的。

#### 异构编程
如图8所示，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，该设备作为运行C程序的主机的协处理器运行。 例如，当内核在GPU上执行而其余的C程序在CPU上执行时就是这种情况。

CUDA编程模型还假设主机和设备都在DRAM中保持它们自己独立的存储空间，分别称为主机存储器和设备存储器。 因此，程序通过调用CUDA运行时（在[编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中描述）来管理内核可见的全局，常量和纹理内存空间。 这包括设备内存分配和释放以及主机和设备内存之间的数据传输。

Unified Memory提供托管内存以桥接主机和设备内存空间。 可以从系统中的所有CPU和GPU访问托管内存，作为具有公共地址空间的单个连贯内存映像。 此功能可实现设备内存的超额预订，并且无需在主机和设备上显式镜像数据，从而大大简化了移植应用程序的任务。 有关统一内存的介绍，请参阅[统一内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd0)。

![](https://note.youdao.com/yws/public/resource/a316463a0e3ab0424a29e5a0c00b2395/xmlnote/3C43B5D6EDB74EB5BD66D8A06EC43A2A/2953)

#### 计算能力
设备的计算能力由版本号表示，有时也称为“SM版本”。 此版本号标识GPU硬件支持的功能，并由运行时的应用程序用于确定当前GPU上可用的硬件功能和（或）指令。

计算能力包括主修订号X和次修订号Y，并由X.Y表示。

具有相同主要版本号的设备具有相同的核心架构。 基于Volta架构的设备的主要版本号为7，基于Pascal架构的设备为6，基于Maxwell架构的设备为5，基于Kepler架构的设备为3，基于Fermi架构的设备为2， 和1用于基于特斯拉架构的设备。

次要修订号对应于核心架构的增量改进，可能包括新功能。

[启用CUDA的GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)列出了所有支持CUDA的设备及其计算功能。 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)提供每种计算能力的技术规范。

>注：特定GPU的计算能力版本不应与CUDA版本(例如CUDA 7.5、CUDA 8、CUDA 9)相混淆，CUDA版本是CUDA软件平台的版本。应用程序开发人员使用CUDA平台创建在许多代GPU体系结构上运行的应用程序，包括尚未发明的未来GPU体系结构。虽然新版本的CUDA平台通常通过支持新GPU体系结构的计算能力版本来添加对该体系结构的本地支持，但新版本的CUDA平台通常还包含独立于硬件的软件功能。

从CUDA 7.0和CUDA 9.0开始，不再支持Tesla和Fermi架构。

## 编程接口
CUDA C为熟悉C编程语言的用户提供了一条简单的路径，可以轻松编写程序以供设备执行。

它由C语言的最小扩展集和运行时库组成。

