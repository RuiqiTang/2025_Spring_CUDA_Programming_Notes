# CUDA 编程模型笔记

## 1. CPU 与 GPU 的交互 

CPU 和 GPU 具有各自独立的物理内存空间 。它们通过 PCIe 总线进行互连 。这种交互方式会带来较高的开销 。

### CPU 与 GPU 架构对比 

| 特性 | CPU | GPU |
|---|---|---|
| **优化目标** | 为对缓存数据集的低延迟访问进行优化  | 为数据并行、吞吐量计算进行优化  |
| **晶体管分配** | 更多晶体管专用于控制逻辑  | 更多晶体管专用于计算 (ALU)  |
| **延迟处理** | 必须最小化每个线程内的延迟  | 通过其他线程 warp 的计算来隐藏延迟  |
| **核心类型** | 低延迟处理器  | 高吞吐量处理器 (流式多处理器 SM)  |

**补充知识点：**
PCIe (Peripheral Component Interconnect Express) 是一种高速串行计算机扩展总线标准，它允许计算机组件之间进行高速数据传输。CPU 和 GPU 之间的数据传输通常需要通过 PCIe 总线进行显式的数据拷贝，这会引入一定的延迟，因此在 CUDA 编程中，应尽量减少 CPU 与 GPU 之间的数据传输次数。

## 2. GPU 内存层次结构 

GPU 内存具有多级层次结构，访问速度从快到慢，容量从少到多 。

### GPU 内部内存拓扑 

一个 Streaming Multiprocessor (SM) 内部包含：
* **寄存器 (Registers)**： dedicated hardware, 访问速度最快，约1个周期 。每个线程拥有自己的私有寄存器 。
* **共享内存 (Shared Memory)**： dedicated hardware, 约10个周期 。位于 SM 片上，允许同一线程块内的线程高效共享数据，并提供较低的延迟访问 。线程可以通过 `__syncthreads()` 同步对共享内存的访问 。
* **L1 缓存 (L1 Cache)**：位于 SM 片上 。
* **纹理内存 (Texture Memory)**：DRAM, 缓存，10-100个周期，只读 。
* **常量内存 (Constant Memory)**：DRAM, 缓存，10-100个周期，只读 。

### GPU 外部内存拓扑 

* **L2 缓存 (L2 Cache)**：位于芯片外部，速度慢于片上内存 。
* **全局内存 (Global Memory)**：DRAM, 无缓存，访问速度最慢，约100个周期 。所有线程都可以访问全局内存 。
* **局部内存 (Local Memory)**：DRAM, 无缓存，访问速度约100个周期 。

**内存速度总结:**

| 内存类型 | 存储位置 | 访问速度 (大致周期数) | 特性 |
|---|---|---|---|
| 寄存器 (Register) | 专用硬件 | ~1 | 每个线程私有 |
| 共享内存 (Shared Memory) | 专用硬件 (SM片上) | ~10 | 同一线程块内线程共享，需同步 |
| L1 缓存 (L1 Cache) | SM片上 | 变动 | 缓存 |
| 纹理内存 (Texture Memory) | DRAM | 10-100 | 只读，缓存 |
| 常量内存 (Constant Memory) | DRAM | 10-100 | 只读，缓存 |
| 局部内存 (Local Memory) | DRAM | ~100 | 无缓存 |
| L2 缓存 (L2 Cache) | 芯片外部 | 变动 | 缓存 |
| 全局内存 (Global Memory) | DRAM | ~100 | 所有线程可访问，无缓存 |

**补充知识点：**
理解内存层次结构对于优化 CUDA 程序至关重要。频繁访问寄存器和共享内存能够显著提高性能，而访问全局内存应尽量减少，并注意内存访问模式以最大化吞吐量（例如，合并访问）。

## 3. GPU 线程层次结构 

CUDA 编程模型通过线程层次结构来组织大量的并行任务 。

### 线程网格 (Grid) 

* 一个 **内核 (Kernel)** 作为线程网格执行 。
* 网格由多个 **线程块 (Thread Block)** 组成 。
* 所有线程共享全局内存 。
* 网格的尺寸在内核启动时指定 。

### 线程块 (Thread Block) 

* 线程块是线程的批次 。
* 线程块内的线程可以相互协作 。
* **协作方式**：
    * 通过 `__syncthreads()` 同步访问共享内存，避免数据竞争 。
    * 通过低延迟的共享内存高效共享数据 。
* **独立性**：不同线程块的线程不能直接协作，只能通过较慢的全局内存进行通信 。
* **调度**：线程块可以以任何顺序并行或串行地由任何 SM 进行调度 。SM 的数量是可扩展的 。

### 线程 (Thread) 

* 一个内核有数千个线程 。
* 线程由 CUDA 核心执行 。
* 每个线程具有唯一的 ID 。

### 内置变量和维度 

* **`dim3 grid(Gx, Gy, Gz), block(Bx, By, Bz);`**：在内核启动时指定网格和块的尺寸 。
    * `Gx, Gy, Gz` 表示网格中块的数量。
    * `Bx, By, Bz` 表示每个块中线程的数量。
* **`kernel<<<grid, block>>>(...);`**：内核启动语法 。
* **`threadIdx.[x y z]`**：线程块内的线程索引 。
* **`blockIdx.[x y z]`**：网格内的块索引 。
* **`blockDim.[x y z]`**：每个块中的线程数 。
* **`gridDim.[x y z]`**：网格中的块数 。

**代码样例：**

```c++
// 定义一个二维的网格和块
dim3 gridSize(16, 16); // 网格包含 16x16 个线程块
dim3 blockSize(32, 32); // 每个线程块包含 32x32 个线程

// 启动 CUDA 内核
myKernel<<<gridSize, blockSize>>>(d_data);

// 在核函数内部获取线程的全局索引 (一维示例)
__global__ void myKernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程在整个网格中的一维索引
    // ... 对 data[idx] 进行操作 ...
}

// 获取二维线程索引
__global__ void my2DKernel(float *data, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col; // 计算当前线程在二维数据中的索引
    // ... 对 data[idx] 进行操作 ...
}
```

## 4. GPU 程序模型 (SIMT) 

GPU 采用 **SIMT (Single Instruction, Multiple Thread)** 架构，它是 SIMD (Single Instruction, Multiple Data) 的 GPU 版本 。

### SIMT 特性 

* **高并行性**：通过数千个线程实现高并行性 。
* **延迟隐藏**：大规模的交织 (interleaving) 用于隐藏内存访问延迟 。
* **指令流**：多个线程执行相同的指令流 。
* **独立性**：
    * 每个线程有自己的指令地址计数器 。
    * 每个线程有自己的寄存器状态 。
    * 每个线程可以有独立的执行路径 。
* **调度**：数千个线程在 GPU 上托管和调度 。

### GPU 执行模型 

* 线程块彼此独立执行 。
* 它们可以以任何顺序（并行或串行）执行 。
* 它们可以由任何 SM 以任何顺序调度 。
* SM 的数量是可扩展的 。

## 5. CUDA：Extended C 

CUDA 扩展了 C 语言，允许开发者编写在 GPU 上执行的并行代码 。

### 限定符 (Qualifiers) 

| 限定符 | 作用域 (Scope) | 生命周期 (Life Span) | 调用位置 (Call Station) | 执行位置 (Execution Station) |
|---|---|---|---|---|
| `__global__` | 网格 (Grid) | 应用程序 (Application) | CPU | GPU |
| `__device__` | 设备 (Device) | 设备 (Device) | 设备 (Device) | 设备 (Device) |
| `__host__` | 主机 (Host) | 主机 (Host) | 主机 (Host) | 主机 (Host) |
| `__shared__` | 线程块 (Block) | 线程块 (Block) | N/A | GPU |
| `__constant__` | 网格 (Grid) | 应用程序 (Application) | N/A | GPU |

**`__global__` 函数 (Kernel):**
* 定义一个内核函数 。
* 在 CPU 上调用，在 GPU 上执行 。
* 必须是 `void` 返回类型 。
* 只能访问设备内存 。
* 不支持可变数量的参数 。

**`__device__` 函数:**
* 可以在 GPU 上调用，也在 GPU 上执行 。
* 通常用于辅助核函数的功能。

**`__host__` 函数:**
* 可以在 CPU 上调用，也在 CPU 上执行 。
* 这是标准的 C/C++ 函数。

### 内置变量 (Built-in Variables) 

* `threadIdx`: 线程块内的线程索引 。
* `blockIdx`: 网格内的块索引 。
* 还有 `blockDim` 和 `gridDim` 用于获取块和网格的维度 。

### 内在函数 (Intrinsics) 

* `__syncthreads()`: 同步同一线程块内的所有线程 。在访问共享内存后必须使用，以确保所有线程都已完成其共享内存访问 。

### 运行时 API (Runtime API) 

提供用于内存管理、符号管理和执行管理的功能 。
* **内存管理**：
    * `cudaMalloc()`: 在 GPU 上分配内存 。
    * `cudaFree()`: 释放 GPU 内存。
    * `cudaMemcpy()`: 在主机和设备之间，或设备内存之间传输数据 。
        * `cudaMemcpyHostToDevice`: 从主机到设备。
        * `cudaMemcpyDeviceToHost`: 从设备到主机。
        * `cudaMemcpyDeviceToDevice`: 设备内部传输。
* **符号管理**：
    * `cudaMemcpyToSymbol()`: 将数据传输到 `__device__` 或 `__constant__` 限定的全局变量（符号）中 。注意：直接使用变量的地址会报错，需要直接传入变量名或通过 `cudaGetSymbolAddress` 获取地址 。
    * `cudaGetSymbolAddress()`: 获取 `__device__` 或 `__constant__` 限定变量的设备地址 。

**代码样例：**

```c++
// 设备全局变量声明
__device__ float devDataGlobal; 

// 内核函数定义
__global__ void convolve (float *image) { 
    // 共享内存声明
    __shared__ float region[M]; 
    // ...
    region[threadIdx.x] = image[threadIdx.x]; // 访问共享内存
    __syncthreads(); // 同步线程块内的线程 
    // ...
    image[threadIdx.x] = devDataGlobal; // 访问设备全局变量
}

// 主机端代码
int main() {
    float *h_image; // 主机内存指针
    void *d_image;  // 设备内存指针
    int bytes = 1024 * sizeof(float); // 示例大小

    // 在主机上分配内存
    h_image = (float*)malloc(bytes);
    // 初始化 h_image ...

    // 在 GPU 上分配内存
    cudaMalloc(&d_image, bytes); 

    // 将数据从主机传输到设备
    cudaMemcpy(d_image, h_image, bytes, cudaMemcpyHostToDevice);

    // 将值传输到 __device__ 变量 (正确方式)
    float value = 123.45f;
    cudaMemcpyToSymbol(devDataGlobal, &value, sizeof(float)); // OK 

    // 或者通过 cudaGetSymbolAddress 获取地址后使用 cudaMemcpy
    float *dptr = NULL;
    cudaGetSymbolAddress((void**)&dptr, devDataGlobal);
    cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice); // OK 

    // 启动内核函数
    // 100个块，每个块10个线程
    convolve<<<100, 10>>>(d_image); 

    // 将结果从设备传输回主机 (如果需要)
    cudaMemcpy(h_image, d_image, bytes, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_image);
    free(h_image);

    return 0;
}
```