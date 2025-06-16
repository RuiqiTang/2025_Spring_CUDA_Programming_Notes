# CUDA 并行规约优化笔记

本笔记总结了 NVIDIA 开发者技术工程师 Mark Harris 介绍的 CUDA 并行规约优化策略，通过逐步讲解 7 个不同版本的规约内核，展示了从基础实现到高度优化的过程。

## 1. 并行规约概述

并行规约是一种常见且重要的数据并行原语，用于将一组数据通过某种操作（如求和、求最大值、求最小值等）归约为单个结果。

* **实现难度**：在 CUDA 中实现规约相对容易，但要实现高性能的规约则更为困难。
* **优化范例**：规约是一个极佳的优化范例，可以演示多种重要的 CUDA 优化策略。
* **树状规约**：在单个线程块内部，通常采用基于树的方法进行规约，每个线程处理一部分数据，然后逐步向上汇总。

## 2. 跨线程块的规约问题与解决方案

### 2.1 问题：全局同步

**挑战**：对于非常大的数组，需要使用多个线程块来处理，并使 GPU 上的所有多处理器保持忙碌。每个线程块规约数组的一部分，但如何在线程块之间传递部分结果？

**自然想法**：如果在所有线程块之间存在全局同步，那么可以轻松地递归地规约非常大的数组。

**CUDA 现状**：CUDA **没有**硬件实现的全局同步机制。

**原因**：
* **硬件成本高昂**：对于具有大量处理器的 GPU 来说，在硬件中构建全局同步成本非常高。
* **效率限制**：全局同步会强制程序员运行更少的线程块（不能超过 `多处理器数量 * 每个多处理器驻留块数量`）以避免死锁，这可能会降低整体效率。

**解决方案**：**分解为多个内核**。

* **同步点**：每次内核启动都充当一个全局同步点。
* **开销**：内核启动的硬件开销可以忽略不计，软件开销较低。

### 2.2 问题：多内核的开销

**挑战**：虽然每次内核启动的开销相对较低（数量级为毫秒），但对于少于数百万个元素的数组，这种开销可能会抵消并行化带来的好处。

**解决方案**：**单个内核，多个线程块**。

* **方法**：
    1.  将整个数组的规约任务分配给多个线程块。
    2.  每个线程块将其部分结果写入全局内存。
    3.  通过一系列内核启动（而不是一个大内核的全局同步），递归地规约这些部分结果，直到得到最终结果。
* **优势**：这种方法比执行多个内核启动要快得多，因为避免了频繁的内核启动开销。
* **结果识别**：每个线程块需要知道它在全局内存中的哪个位置写入结果。使用 `blockIdx.x` 为每个块指定一个唯一的输出位置：
    `g_odata[blockIdx.x] = reduction_result_of_this_block;`

## 3. 规约内核优化版本迭代

本节将逐步介绍不同版本的规约内核及其优化点。

### 3.1 初步内核 1

* **功能**：每个线程块规约其分配的数组部分，每个线程加载两个元素并求和，然后通过共享内存树状规约。
* **共享内存**：`extern __shared__ float sdata[];`
* **加载数据**：
    ```c++
    int tid = threadIdx.x;
    int i = blockIdx.x * blockSize * 2 + tid; // 每个线程处理两个元素
    sdata[tid] = g_idata[i] + g_idata[i+blockSize]; // 初步求和并存入共享内存
    __syncthreads(); // 确保所有数据加载完成
    ```
* **线程块内规约**：
    ```c++
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // 每次迭代后同步
    }
    ```
* **结果输出**：
    ```c++
    if (tid == 0) { // 只有线程0将结果写入全局内存
        g_odata[blockIdx.x] = sdata[0];
    }
    ```
* **性能（示例）**：22.80 毫秒 (对于 1000 万个元素数组)

### 3.2 优化 #1：避免发散分支 (内核 2)

* **问题**：在 `for` 循环中的 `if (tid < s)` 语句会导致发散分支。
    * **SIMT 特性**：CUDA 的 warp（通常 32 个线程）中的所有线程必须执行相同的指令。
    * **发散**：如果 `if` 语句中的所有线程不在同一路径上执行，warp 会按顺序执行每个分支，禁用在另一个分支上的线程，导致性能损失。
    * 当 `s < warpSize` 时，一个 warp 中的线程会发散（例如，当 `s = 16` 时，`tid < 16` 的线程进入分支，`tid >= 16` 的线程不进入）。
* **解决方案**：**交错寻址 (Interleaved Addressing)**。
    * 更改循环中的加法操作，使得所有线程都执行加法，从而避免 `if (tid < s)`。
    * **代码修改**：
        ```c++
        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
            // 所有线程都执行加法，避免发散分支
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        ```
* **性能提升**：19.38 毫秒 (约 15% 提升)。

### 3.3 优化 #2：避免 Bank 冲突 (内核 3)

* **问题**：共享内存被分成称为“bank”的 32 位模块，可以同时访问。如果一个 warp 中的多个线程尝试访问同一个 bank，则访问将被序列化，导致 bank 冲突。
    * 在交错寻址版本中，`sdata[tid] += sdata[tid + s];` 可能导致 bank 冲突。
    * 当 `s` 为 `warpSize` (32) 的倍数时，例如 `s = 32`，`sdata[tid]` 和 `sdata[tid + 32]` 将会访问相同的 bank，导致 bank 冲突。
* **解决方案**：**顺序寻址 (Sequential Addressing) 或混合寻址策略**。
    * 对于 `s >= 32`（即 `s` 为 32 的倍数），使用交错寻址，因为此时 `tid` 和 `tid+s` 所在的 bank 不会冲突（或者即使冲突，也是不同 warp 之间的冲突，不会导致序列化）。
    * 对于 `s < 32`（即 `s` 不是 32 的倍数），切换到不同的寻址模式以避免 bank 冲突，通常是使 `tid` 和 `tid+1` 访问不同的 bank。一种简单的方法是**将共享内存大小加倍，并写入 `sdata[2*tid]`**，这样 `sdata[2*tid]` 和 `sdata[2*tid+1]` 永远不会在同一个 bank 中。
    * **代码修改**：
        ```c++
        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
            if (s >= 32) { // s 较大时，使用交错寻址 (通常不会导致 bank 冲突)
                sdata[tid] += sdata[tid + s];
            } else { // s 较小时，切换到顺序寻址，消除 bank 冲突
                sdata[tid] += sdata[tid + 1]; // 注意：这里需要 sdata 预先被组织成避免bank冲突的模式，例如2*tid
            }
            __syncthreads();
        }
        ```
        *注：原始PDF中内核3的代码示例可能与文字描述略有出入，这里根据描述进行调整。更准确的避免bank冲突的经典做法是填充共享内存，例如对于大小为 `N` 的共享内存数组，可以声明为 `float sdata[N + N / NUM_BANKS]`，并在访问时使用 `sdata[tid + tid / NUM_BANKS]`。但在规约场景中，通过调整 `s` 的大小来选择不同的寻址模式是常见的优化手法。*
* **性能提升**：18.06 毫秒 (约 7% 提升)。

### 3.4 优化 #3：在全局加载期间首次添加 (内核 4)

* **问题**：每个线程块需要加载 `2 * blockSize` 个元素。初步内核中，每个线程加载两个元素并执行一个加法 `sdata[tid] = g_idata[i] + g_idata[i + blockSize];`。
* **优化思路**：将第一次加法操作提前到数据从全局内存加载到共享内存的过程中。这避免了在规约循环中进行额外的共享内存读取和写入。
* **代码修改**：
    ```c++
    // 加载两个元素并立即相加，存入共享内存
    sdata[tid] = g_idata[i] + g_idata[i + blockSize];
    __syncthreads();
    ```
    *注：这个优化在内核1中已经体现，但PDF强调其作为一个优化点再次提及。它实际上是利用了加载操作的带宽，并减少了后续共享内存操作的次数。*
* **性能提升**：17.50 毫秒 (约 3% 提升)。

### 3.5 优化 #4：展开最后一个 Warp (内核 5)

* **问题**：`__syncthreads()` 调用可能很昂贵，因为它强制所有线程等待。在规约循环的最后几步，当 `s < warpSize` 时，一个 warp 内的所有线程都参与计算。
* **优化思路**：展开循环的最后一步，使其不再需要 `__syncthreads()` 调用。这是因为在一个 warp 内部，线程是隐式同步的（单指令多线程），不需要显式同步。
* **代码修改**：
    ```c++
    // 循环规约直到 s = 32
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (s >= 32)
            sdata[tid] += sdata[tid + s];
        else // This branch is theoretically dead if loop condition is s > 32
            sdata[tid] += sdata[tid + 1];
        __syncthreads();
    }

    // 展开的 warp 规约 (s < 32 时，不再需要 __syncthreads())
    // 假设 warpReduce 是一个辅助函数，在 warp 内部执行规约
    if (tid < 32) {
        // warpReduce(sdata, tid); // 假设 warpReduce 负责如下操作
        if (tid < 16) sdata[tid] += sdata[tid + 16];
        if (tid < 8) sdata[tid] += sdata[tid + 8];
        if (tid < 4) sdata[tid] += sdata[tid + 4];
        if (tid < 2) sdata[tid] += sdata[tid + 2];
        if (tid < 1) sdata[tid] += sdata[tid + 1];
    }
    ```
* **性能提升**：16.74 毫秒 (约 4% 提升)。

### 3.6 优化 #5：完全展开 (内核 6)

* **问题**：循环本身也有开销。
* **优化思路**：对于固定的 `blockSize`（例如 512），规约循环的迭代次数是固定的 (9 次)。可以完全展开循环，消除循环开销。
* **代码修改**：
    ```c++
    // 完全展开循环
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    // 接下来的小于32的部分可以通过 warpReduce 辅助函数或继续展开
    if (tid < 32) warpReduce(sdata, tid); // 假设 warpReduce 包含了剩余的展开逻辑
    ```
* **性能提升**：16.54 毫秒 (约 1% 提升)。

### 3.7 优化 #6：每个线程处理多个元素 (内核 7 - 最终优化内核)

* **问题**：在之前的内核中，每个线程只处理两个初始元素。对于较大的数组，这意味着可能需要更多的线程块，或者线程块内部的线程利用率不高。
* **优化思路**：通过让每个线程处理更多元素来进一步优化。这允许我们用更少的线程实现相同的规约，从而减少共享内存的使用（如果每次迭代处理的元素总数不变）并可能提高全局内存访问的效率（更好的 coalescing）。
* **代码修改**：
    ```c++
    // 假设每个线程处理4个元素
    int tid = threadIdx.x;
    // 调整索引以读取更多数据
    int i = blockIdx.x * (blockSize * ELEMENTS_PER_THREAD) + tid;
    // 加载更多元素并初步相加
    sdata[tid] = g_idata[i] + g_idata[i + blockSize] + g_idata[i + blockSize * 2] + g_idata[i + blockSize * 3]; // 示例：处理 4 个元素
    __syncthreads();
    ```
    *注：这个优化通常会在初始加载阶段完成，即每个线程加载并求和其负责的多个全局内存元素，然后将这个部分和存入共享内存的 `sdata[tid]`。后续的规约逻辑保持不变。这需要相应地调整内核启动时的 `gridDim.x` 和 `blockSize`。*
* **性能提升**：PDF中未给出具体数字，但通常会带来显著提升，尤其是在全局内存访问效率方面。

## 4. 总结与最终优化内核

最终的优化内核结合了上述所有优化策略：

1.  **在加载时首次添加**：每个线程加载两个元素并立即相加。
2.  **避免发散分支**：使用交错寻址或确保 `if` 条件不会导致 warp 内线程发散。
3.  **避免 Bank 冲突**：根据 `s` 的大小选择合适的寻址模式，或通过填充共享内存来消除冲突。
4.  **展开最后一个 Warp**：消除 `s < 32` 时循环内部的 `__syncthreads()` 调用。
5.  **完全展开**：消除循环本身的开销。
6.  **每个线程处理多个元素**：提高线程利用率和全局内存访问效率。

```c++
// 最终优化内核示例 (结合了多种优化思路)
extern __shared__ float sdata[]; // 共享内存，大小需要根据 blockSize 和填充策略设置

// 辅助函数，在单个 warp 内执行规约 (不需要 __syncthreads())
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reductionKernel(float* g_idata, float* g_odata, unsigned int N) {
    unsigned int tid = threadIdx.x;
    // 调整索引以处理更多元素 (例如每个线程处理 2 或 4 个元素)
    // 这里的 blockSize 是线程块大小，每个块处理 2*blockSize 个元素
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    // 1. 在加载时首次添加
    // 确保 i + blockDim.x 在数组边界内
    sdata[tid] = (i < N ? g_idata[i] : 0.0f) + ((i + blockDim.x) < N ? g_idata[i + blockDim.x] : 0.0f);
    __syncthreads(); // 确保所有共享内存加载完成

    // 2. 规约循环 (避免发散分支和 Bank 冲突)
    // 3. 部分展开循环 (例如直到 s=64 或 s=32)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) { // 避免发散分支，但确保 bank 冲突管理得当
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 4. 展开的 warp 规约 (当 s < 32 时，利用 warp 内部的隐式同步)
    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    // 5. 只有线程 0 将最终结果写入全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 主机端调用示例
void host_reduction(float* h_idata, float* h_odata, unsigned int numElements) {
    // ... (内存分配和数据传输)

    // 第一次规约：每个块规约 2*blockSize 个元素
    unsigned int blockSize = 512; // 示例块大小
    unsigned int numBlocks = (numElements + (blockSize * 2 - 1)) / (blockSize * 2);
    reductionKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_idata, d_odata_partial, numElements);
    cudaDeviceSynchronize(); // 等待所有块完成

    // 递归规约，直到只剩一个元素
    while (numBlocks > 1) {
        numElements = numBlocks; // 更新待规约元素数量为上一次的部分和数量
        numBlocks = (numElements + (blockSize * 2 - 1)) / (blockSize * 2); // 重新计算块数量
        reductionKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_odata_partial, d_odata_partial, numElements);
        cudaDeviceSynchronize();
    }

    // ... (结果复制回主机)
}
```

## 5. 性能比较总结

PDF 中提供了不同优化版本的性能对比图，展示了随着优化程度的增加，执行时间逐渐减少。

* **内核 1**：基础交错寻址，存在发散分支。
* **内核 2**：修正了发散分支。
* **内核 3**：引入顺序寻址以避免 Bank 冲突。
* **内核 4**：首次加法在全局加载期间完成。
* **内核 5**：展开最后一个 warp。
* **内核 6**：完全展开。
* **内核 7**：每个线程处理多个元素 (未在图中明确标出，但通常是最终的性能提升点)。

**关键优化点**：
* **全局内存访问模式**：尽量实现合并访问 (Coalescing)，减少加载次数。
* **共享内存访问模式**：避免 Bank 冲突。
* **线程同步**：减少 `__syncthreads()` 调用，尤其是在 warp 内部利用隐式同步。
* **分支发散**：避免 `if` 语句导致 warp 内线程发散。
* **循环开销**：通过展开循环来消除。

通过这些细致的优化，可以显著提升 CUDA 并行规约的性能，尤其对于大规模数据集。