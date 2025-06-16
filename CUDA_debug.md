## CUDA 调试与分析笔记

### 1. CUDA 工具概述

* **开发工具**: `NVCC`, `cuobjdump` 
* **调试工具**: `cuda-gdb`, `cuda-memcheck`, API 错误检查 
* **性能分析工具**: `Nsight Systems`, `Nsight Compute` 
* **系统管理工具**: `nvidia-smi` 

### 2. 并行线程执行 (PTX)

PTX (Parallel Thread Execution) 是 NVIDIA CUDA 架构的虚拟指令集 。

#### NVCC 两步构建模型

`NVCC` 编译器采用两步构建模型将 `.cu` 设备代码编译为最终的可执行文件 ：

1.  **阶段 1 (PTX 生成)**: 将 `.cu` 设备代码编译成 `x.ptx` 文件，这是针对虚拟计算架构的 PTX 代码 。
2.  **阶段 2 (Cubin 生成)**: 将 `x.ptx` 文件编译成 `x.cubin` 文件，这是针对真实 SM (Streaming Multiprocessor) 架构的机器码 。

#### 兼容性考虑

* **二进制兼容性**: 通常在相同架构的 GPU 上得到保证，但不同架构之间通常不保证 。
* **PTX 确保兼容性**: PTX 确保了不同代 GPU 之间的二进制兼容性 。这意味着为旧架构编译的 PTX 代码可以在新架构的 GPU 上通过 JIT (Just-In-Time) 编译运行 。

#### PTX 原理与用法

* `--gpu-architecture (-arch) compute_xx`: 指示用于生成相应 PTX 代码的虚拟架构 。
* `--gpu-code (-code) sm_xx`: 指示用于生成相应机器码的实际硬件架构 。

**示例**:
* `-arch=compute_20, -code=sm_20`: PTX 对应 `compute_20`，生成 `sm_20` 架构的机器码。如果硬件是 `sm_30`，PTX 会在运行时 JIT 编译为 `sm_30` 代码 。
* `-arch=sm_30` 等同于 `-arch=compute_30, -code=sm_30` 。
* `-arch=compute_30, -code=sm_20` 是非法的 。

**补充知识点：JIT 编译**
Just-In-Time (JIT) 编译是一种在程序运行时而非编译时将代码编译成机器码的技术。在 CUDA 中，当为特定虚拟架构（例如 `compute_20`）生成的 PTX 代码在支持不同物理架构（例如 `sm_30`）的 GPU 上运行时，CUDA 驱动程序会执行 JIT 编译，将 PTX 代码转换为目标 GPU 的本地机器码。这确保了 CUDA 程序的向后兼容性。

### 3. 开发工具

#### 3.1 NVCC: CUDA (.cu) 编译器驱动程序

`nvcc` 负责将源程序分离为 CPU (host) 代码和 GPU (device) 代码，并调用相应的工具完成编译和链接。

**常用编译选项**:
* `-Xcompiler <arg>`: 指定 GCC 参数，例如 `-Xcompiler -fopenmp`。
* `-G`: 生成可调试的设备代码。
* `-lineinfo`: 生成行号信息。
* `-arch=sm_xx`: 指定设备架构，例如 `-arch=sm_80` 。
* `-maxrregcount <num>`: 限制每个线程的最大寄存器数量 。
* `--ptxas-options=-v`: 直接指定选项给 PTX 优化汇编器 `ptxas` 。

**代码样例：NVCC 编译命令**
```bash
nvcc my_cuda_program.cu -o my_cuda_program -G -lineinfo -arch=sm_80
```
这个命令会编译 `my_cuda_program.cu` 文件，生成名为 `my_cuda_program` 的可执行文件，并包含调试信息和行号信息，目标 GPU 架构为 `sm_80`。

#### 3.2 cuobjdump

`cuobjdump` 类似于 `objdump`，用于提取 CUDA 二进制文件（`.cubin` 或 host binary）的信息 。

**常用命令**:
* `cuobjdump -sass <file>`: 反汇编 CUDA 二进制文件 ，显示 SASS (Streaming Assembler) 代码。
* `cuobjdump -ptx <file>`: 提取 PTX 代码 。
* `cuobjdump -elf <file>`: 以可读形式获取目标文件的 ELF 段 。

**代码样例：cuobjdump 使用**
```bash
nvcc my_kernel.cu -o my_kernel.cubin -arch=sm_80 --ptx
cuobjdump -sass my_kernel.cubin
cuobjdump -ptx my_kernel.cubin
```
上述命令首先编译一个 CUDA 内核到 `.cubin` 文件，然后使用 `cuobjdump` 查看其 SASS 和 PTX 代码。

### 4. 调试工具

#### 4.1 cuda-gdb

`cuda-gdb` 是 Linux 和 Mac 平台上用于调试 CUDA 程序的工具 ，它是 GDB 的扩展，支持 CPU 和 GPU 代码的调试 。

**启动与运行**:
* `cuda-gdb ./a.out` 
* `(cuda-gdb) run (r) [arguments]` 
* `(cuda-gdb) next (n)`: 单步执行 
* `(cuda-gdb) continue (c)`: 继续运行 
* `(cuda-gdb) kill`: 终止程序 
* `(cuda-gdb) CTRL+C`: 停止调试 

**断点设置**:
* **符号断点**: `(cuda-gdb) break (b) function` 或 `(cuda-gdb) break (b) class::method` 
* **行断点**: `(cuda-gdb) break (b) file.cu::line-number` 
* **地址断点**: `(cuda-gdb) break (b) address` 
* **内核入口断点**: `(cuda-gdb) set cuda break_on_launch application` 
* **条件断点**: 支持 C/C++ 条件表达式和 GPU 内置变量 (如 `blockIdx`, `threadIdx`) 。
    * 例如: `(cuda-gdb) break file.cu:23 if threadIdx.x == 1 && i < 5` 

**焦点切换**:
`cuda-gdb` 允许在主机线程和 GPU 线程之间切换焦点 。GPU 线程的焦点可以通过软件坐标 (线程、块、内核) 和硬件坐标 (lane, warp, sm, device) 来指定 。
* 查看当前焦点: `(cuda-gdb) cuda device sm warp lane block thread` 
* 切换焦点: `(cuda-gdb) cuda device 0 sm 1 warp 2 lane 3` 

**补充知识点：GPU 线程层次结构**
CUDA 程序的执行是高度并行的，其线程组织成一个层次结构：
* **Kernels**: 在 GPU 上执行的函数。
* **Grid**: 一组线程块。
* **Block (Thread Block)**: 一组线程，在同一个 SM 上协作执行。块内的线程可以通过共享内存进行通信和同步。
* **Thread**: 最基本的执行单元。
* **Warp**: 由 32 个线程组成的组，这些线程以 SIMT（Single Instruction, Multiple Thread）方式执行相同的指令。一个块由一个或多个 warp 组成。
* **Lane**: 指 warp 中的一个线程。

数学公式:
* 一个 CUDA 块中的线程总数可以表示为: $ThreadsPerBlock = dimBlock.x \times dimBlock.y \times dimBlock.z$
* 一个 CUDA 网格中的块总数可以表示为: $BlocksPerGrid = dimGrid.x \times dimGrid.y \times dimGrid.z$

#### 4.2 cuda-memcheck

`cuda-memcheck` 是一个命令行内存管理工具，用于检测 CUDA 程序的内存错误。

**用法**: `cuda-memcheck [options] ./a.out [app_options]`

**编译时选项**:
* 编译时需添加 `-g` 和 `-lineinfo` 选项。
* 添加 `-Xcompiler -rdynamic` 选项以保留函数符号。

**检测错误类型**:
* 访问冲突 (`Access Violation Fault`)
* 硬件异常 (`Hardware Abnormal`)
* `malloc`/`free` 错误 (内核内部)
* 内存泄漏 (`Memory Leak`)
* CUDA API 错误 (`CUDA API Error`)

**支持的模式**:
`cuda-memcheck` 支持多种模式来检测不同类型的错误：
* **Memcheck**: 内存访问错误和内存泄漏检测。
* **Racecheck**: 共享内存数据访问危害检测。
* **Initcheck**: 未初始化设备全局内存访问检测。
* **Synccheck**: 线程同步危害检测。

**代码样例：cuda-memcheck 使用**
```bash
nvcc -g -lineinfo -Xcompiler -rdynamic my_cuda_program.cu -o my_cuda_program
cuda-memcheck ./my_cuda_program
```

#### 4.3 CUDA API 错误检查

所有 CUDA API 函数都会在完成后返回运行状态，`cudaSuccess` 表示成功，其他值表示异常 。

**API 查询状态函数**:
* `cudaError_t err = cudaGetLastError()`: 重置错误状态 。
* `cudaError_t err = cudaPeekLastError()`: 不重置错误状态 。

**用户自定义宏**: 通常，用户会定义 `cudaCheckErrors` 宏来封装 API 调用，便于错误检查和报告 [cite: 15, 16]。

**代码样例：用户自定义错误检查宏**
```c++
#define cudaCheckErrors(err) __checkCudaErrors (err, __FILE__, __LINE__) 

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 使用示例
int main() {
    // ...
    cudaCheckErrors(cudaMalloc((void**)&devPtr, size)); // 检查 cudaMalloc 调用是否成功
    // ...
}
```

### 5. 性能分析工具

#### 5.1 Nsight Systems

`Nsight Systems` 用于系统级的应用程序算法调优 ，提供系统范围的性能概览，帮助识别瓶颈。

#### 5.2 Nsight Compute

`Nsight Compute` 专注于调试和优化特定的 CUDA 内核 ，提供详细的性能指标，帮助分析内核效率。

**工作流程**: 通常，会先使用 `Nsight Systems` 进行系统级分析，然后针对发现的性能热点，使用 `Nsight Compute` 深入分析特定的 CUDA 内核 。

### 6. 系统管理工具

#### 6.1 nvidia-smi

`nvidia-smi` 是一个基于 NVML (NVIDIA Management Library) 开发的系统管理工具。

**功能**:
* 查询或修改 GPU 访问模式。
* 查询或重置 ECC 状态。
* 查询 GPU 使用情况。
* 更改 GPU 运行频率和电压 (GPU Boost)。

**输出格式**: 可以是 XML 或文本文件，支持标准输出或重定向输出 [cite: 20]。

**代码样例：nvidia-smi 使用**
```bash
nvidia-smi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
nvidia-smi -L # 列出所有 GPU
```