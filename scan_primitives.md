`Scan Primitives for GPU Computing`这篇论文主要探讨了在GPU上实现Scan（前缀和）原语及其变体（如分段Scan）的方法，并展示了它们在并行算法中的应用。以下是整理的笔记及知识点补充：

## 1. 引言与动机 

  * **GPU的并行能力**：到2007年底，处理器芯片上的晶体管数量已超过十亿，这使得GPU不仅能处理图形应用，还能胜任各种计算密集型通用问题，其性能可达数百GFLOPS。 
  * **GPU高性能的原因**：GPU是一个高度并行的机器。例如，NVIDIA最新的旗舰GPU拥有128个处理器，通过处理数千个并行计算线程来保持这些处理器的繁忙。 
  * **图形编程模型与通用计算**：图形工作负载非常适合细粒度并行工作，图形管线的可编程部分对图元（顶点和片段）进行操作，每个典型帧中都有数十万到数百万的此类图元，每个图元程序都会生成一个线程以保持并行处理器满载。 
  * **数据并行性**：OpenGL和DirectX API中的片段处理是完全数据并行的，片段处理器迭代每个输入片段，为每个片段产生固定数量的输出，这些输出仅依赖于输入的片段（Lefohn等人称之为“单访问迭代器”）。  这种显式数据并行性使得最近的GPU能够有效利用如此多的并行处理器。 
  * **流编程模型**：显式并行性非常适合流编程模型，其中一个程序（内核）并行操作每个输入元素，为每个输入元素产生一个输出元素。图形管线的可编程片段和顶点部分精确匹配这种严格的流编程模型。 
  * **复杂操作的需求**：许多有趣的问题需要更复杂的访问模式，不仅仅是单访问或邻域访问迭代器能支持的。  前缀和（prefix-sum）算法是许多具有复杂访问需求的并行应用中常见的算法模式。 
  * **前缀和 (Prefix-Sum) 介绍**：
      * **定义**：前缀和的输入是一个数值数组，输出是一个等长的数组，其中每个元素是输入数组中所有前面元素的总和。 
      * **示例**：
          * 输入: `[3 1 7 0 4 1 6 3]` 
          * 输出: `[0 3 4 11 11 14 16 22]`  (这是独占前缀和的例子，即输出不包含当前元素自身)
      * **并行计算的挑战**：对于串行处理器，计算前缀和是微不足道的，但对于并行处理器则更困难。  简单地并行计算每个输出的朴素方法是让输出流中的每个元素都对输入流中所有前面的值求和，这需要 $O(n^2)$ 的总内存访问和加法操作，成本过高。 

## 2. Scan：一种高效的并行原语 

  * **Scan原语**：本文通过Scan原语家族解决需要全局输入知识的并行问题。 
  * **Scan的起源和GPU上的发展**：
      * 最早由APL提出 ，并由Blelloch在Connection Machine上推广为基本原语 。
      * Horn首次将其引入GPU用于“非均匀流压缩” 。
      * Hensley等人改进了Horn的实现，但其串行工作复杂度仍为 $O(n \\log n)$ 。
      * Sengupta等人和Greß等人于次年展示了首个 $O(n)$ 的基于GPU的Scan实现 。
  * **Scan操作的类型**：
      * **输入**：数据元素向量和一个带有单位元 $i^{\\dagger}$ 的结合二元函数。  （常见的二元函数包括加法、最小值、最大值、逻辑与、逻辑或等。） 
      * **独占Scan (Exclusive Scan)**：如果输入是 $[a\_0, a\_1, a\_2, a\_3, ...]$，独占Scan产生输出 $[i, a\_0, a\_0 \\oplus a\_1, a\_0 \\oplus a\_1 \\oplus a\_2, ...]$。 
      * **包含Scan (Inclusive Scan)**：如果输入是 $[a\_0, a\_1, a\_2, a\_3, ...]$，包含Scan产生输出 $[a\_0, a\_0 \\oplus a\_1, a\_0 \\oplus a\_1 \\oplus a\_2, a\_0 \\oplus a\_1 \\oplus a\_2 \\oplus a\_3, ...]$。 
      * **实现方式**：本文实现的是独占和Scan作为基本原语，并通过将输入向量加到独占输出向量来生成包含Scan。 
      * **反向Scan**：通过在Scan开始时反转输入元素来支持反向Scan。 

### 2.1. CUDA环境与硬件特性 

  * **NVIDIA CUDA**：作者选择NVIDIA CUDA GPU计算环境进行实现，CUDA提供了一个直接的、通用的C语言接口，用于NVIDIA 8系列GPU上的可编程处理器。 
  * **CUDA的优势**：
      * **通用加载-存储内存架构**：允许GPU程序进行任意的gather和scatter内存访问。 
      * **片上共享内存 (Shared Memory)**：GPU上的每个多处理器包含一个快速片上内存 (NVIDIA 8系列GPU上为16 KB)。在同一多处理器上运行的所有线程都可以从该内存中加载和存储数据。 
      * **线程同步 (Thread Synchronization)**：提供了一个barrier指令，用于GPU多处理器上所有活动线程之间的同步。结合共享内存，此功能允许线程协同计算结果。 
  * **GPU架构**：NVIDIA 8系列GPU具有多个物理多处理器，每个多处理器都有共享内存和多个标量处理器（例如，NVIDIA GeForce 8800 GTX有16个多处理器，每个有八个处理器）。 
  * **CUDA编程模型**：CUDA将GPU程序组织成并行线程块，每个块最多512个SIMD并行线程。  程序员指定线程块的数量和每个块的线程数，硬件和驱动程序将线程块映射到GPU上的并行多处理器。  在一个线程块内，线程可以通过共享内存进行通信，并通过共享内存和线程同步进行协作。 
  * **效率**：高效的CUDA程序既利用了线程块内的线程并行性，也利用了跨线程块的更粗粒度的块并行性。  由于只有同一块内的线程才能通过共享内存和线程同步进行协作，程序员必须将计算划分为多个块。  这种模型虽然增加了编程复杂性，但带来了巨大的性能提升（例如，比OpenGL实现快7倍）。 

### 2.2. $O(n)$ 非分段Scan算法 

  * **实现细节**：本文使用的 $O(n)$ 非分段CUDA Scan实现遵循Blelloch的工作高效公式  和Sengupta等人的GPU实现 ，但针对CUDA进行了效率优化。 
  * **两个阶段**：工作高效的Scan算法需要对数组进行两次遍历：
    1.  **Reduce (上扫)**：自下而上遍历二叉树，计算每个内部节点的局部和。 
    2.  **Down-sweep (下扫)**：自上而下遍历二叉树，利用Reduce阶段计算出的局部和进行最终计算。 
  * **时间复杂度**：每个阶段都需要 $\\log n$ 个并行步骤，总工作复杂度为 $O(n)$。 
  * **多块处理**：如果元素数量超过单个线程块能处理的最大数量，数组会被划分到多个线程块中，并且部分和树的结果将作为第二级递归Scan的输入。  第二级Scan的每个输出元素会被加到第一级Scan对应块的所有元素中。 

**Reduce (上扫) 阶段算法 (Algorithm 1)** 
该算法计算部分和，类似于二叉树的向上遍历：

```
Algorithm 1: The reduce (up-sweep) phase of a work-efficient parallel unsegmented scan algorithm.
1: for d = 0 to log_2(n) - 1 do
2:   for all k = 0 to n - 1 by 2^(d+1) in parallel do
3:     x[k + 2^(d+1) - 1] <- x[k + 2^d - 1] + x[k + 2^(d+1) - 1]
```

  * **解释**：在每一轮 $d$ 中，并行处理的线程会将其左兄弟节点的值加到自己身上。这里的 `x` 数组存储的是当前的累加和。 $2^{d+1}$ 是步长，意味着每 $2^{d+1}$ 个元素中，右侧的元素会累加左侧 $2^d$ 步长前的元素。

**Down-sweep (下扫) 阶段算法 (Algorithm 2)** 
该算法将计算出的前缀和传播回原始位置：

```
Algorithm 2: The down-sweep phase of a work-efficient parallel unsegmented scan algorithm.
1: x[n - 1] <- 0
2: for d = log_2(n) - 1 down to 0 do
3:   for all k = 0 to n - 1 by 2^(d+1) in parallel do
4:     t <- x[k + 2^d - 1]
5:     x[k + 2^d - 1] <- x[k + 2^(d+1) - 1]
6:     x[k + 2^(d+1) - 1] <- t + x[k + 2^(d+1) - 1]
```

  * **解释**：Down-sweep 从树的根部（即最大的 $d$ 值）开始向下遍历。第1行将数组的最后一个元素设置为0（对于独占Scan）。在循环中，`t` 临时存储左兄弟节点的值，然后将父节点的值赋给左兄弟，最后将 `t` （即原始左兄弟值）加到右兄弟节点上。

### 2.3. 分段Scan (Segmented Scan) 

  * **概念**：分段Scan是Scan原语的推广，允许在输入向量的任意分区（“段”）上进行并行Scan。 
  * **段的标记**：段通过标志（flag）来划分，其中设置的标志表示一个段的第一个元素。 
  * **复杂度与性能**：分段Scan的实现具有与Scan相同的 $O(n)$ 复杂度，且仅比Scan慢三倍。 
  * **应用**：分段Scan可作为各种应用的构建块，这些应用以前无法在GPU上高效实现。 

#### 2.3.1. 工作高效的分段Scan算法 

  * **核心贡献**：本文的贡献在于将非分段Scan算法的Reduce和Down-sweep阶段扩展到高效实现GPU上的分段Scan。 
  * **与传统分段Scan的区别**：
      * Schwartz首次提出分段Scan的概念，但未描述如何使用Scan的平衡树方法实现。 
      * Chatterjee等人的实现  紧密绑定于Cray-MP架构，使用宽向量寄存器将大输入数组切割成小块，在每个块内分段Scan串行运行，并高效利用向量掩码寄存器存储标志。 
  * **分块处理**：当输入向量太长，单个线程块无法在共享内存中处理时，需要将算法分块。  非分段Scan的简单并行化结构不直接适用于分段Scan。 
  * **Reduce阶段（Segmented Reduce）**：
      * 与非分段Scan类似，分段Reduce阶段遍历一个二叉树。 
      * 需要计算数据和头标志的中间结果。 
      * 部分OR标志是两个输入头标志的逻辑OR。 
      * 数据部分和的计算与非分段Scan相同，除非右父节点的头标志被设置，在这种情况下，右父节点的数据元素保持不变。 

**Segmented Reduce (上扫) 阶段算法 (Algorithm 3)** 

```
Algorithm 3: The reduce (up-sweep) phase of a segmented scan algorithm.
x denotes the partial sums and f denotes the partial OR flags.
1: for d = 1 to log_2(n) - 1 do
2:   for all k = 0 to n - 1 by 2^(d+1) in parallel do
3:     if f[k + 2^(d+1) - 1] is not set then
4:       x[k + 2^(d+1) - 1] <- x[k + 2^d - 1] + x[k + 2^(d+1) - 1]
5:     f[k + 2^(d+1) - 1] <- f[k + 2^d - 1] | f[k + 2^(d+1) - 1]
```

  * **解释**：如果右侧元素的标志没有设置（即它们属于同一个段），则进行累加操作。同时，对标志进行逻辑或操作，以维护段的边界信息。

  * **Down-sweep阶段（Segmented Down-sweep）**：

      * 从根部向下遍历树，使用Reduce阶段的部分和。 
      * **关键区别**：右子节点 (z) 不总是设置为其父节点 (x) 的值与其左兄弟 (w) 的旧值之和。 
      * **条件赋值**：
          * 如果x右侧位置的标志在输入标志向量中被设置，则右子节点 (z) 设置为0（单位元）。 
          * 否则，如果树中相同位置存储的部分OR被设置，右子节点 (z) 设置为左兄弟 (w) 的原始值。 
          * 如果两个标志都未设置，右子节点 (z) 接收其父节点 (x) 和左兄弟 (w) 原始值之和。 

**Segmented Down-sweep (下扫) 阶段算法 (Algorithm 4)** 

```
Algorithm 4: The down-sweep phase of a segmented scan algorithm.
1: x[n - 1] <- 0
2: for d = log_2(n) - 1 down to 0 do
3:   for all k = 0 to n - 1 by 2^(d+1) in parallel do
4:     t <- x[k + 2^d - 1]
5:     x[k + 2^d - 1] <- x[k + 2^(d+1) - 1]
6:     if fi f_i[k + 2^d] is set then
7:       x[k + 2^(d+1) - 1] <- 0
8:     else if f[k + 2^d - 1] is set then
9:       x[k + 2^(d+1) - 1] <- t
10:    else
11:      x[k + 2^(d+1) - 1] <- t + x[k + 2^(d+1) - 1]
12:    Unset flag f[k + 2^d - 1]
```

  * **解释**：此算法根据标志（`f`）的状态，决定如何更新右子节点的值。如果遇到段边界，则从0开始新的前缀和；如果遇到部分OR标志，则保留左兄弟的原始值；否则，像非分段Scan一样累加。最后，取消设置左兄弟的标志。

#### 2.3.2. 分段Scan实现 

  * **标志表示和API**：

      * 段由与输入向量长度相同的头标志向量表示。 
      * 如果输入向量中的一个元素是段的第一个元素，则头标志中的相应条目设置为1，其余为0。 
      * 概念上，头标志位于两个元素之间：当前段的第一个元素和前一个段的最后一个元素。 
      * 对于向后分段Scan，简单地翻转标志是不够的，标志需要向右移动一位。 

  * **标志的有效存储**：

      * 将布尔标志存储为4字节字是空间效率低下的。 
      * 原计划将32个标志打包成一个整数，但CUDA并行线程的读-修改-写语义限制了这种方法。 
      * **替代方案**：每个字节存储一个标志。  为了减少共享内存冲突，以32个连续4字节字为块分配标志，每个字有四个标志，并以4字节间隔交错存储标志，每32个标志循环到块的开头。  (32反映了NVIDIA 8系列GPU的32线程SIMD warp大小。) 
      * **其他标志表示**：Blelloch提出了两种替代方案：段长度向量和头指针向量。  头标志的优点在于直接与每个元素关联，并且大小与数据元素向量相同，易于在线程块之间并行化。 

  * **多块分段Scan**：

      * 与非分段Scan不同，分段Scan结果不能通过统一加法跨块传播。 
      * 加法操作仅作用于每个块的第一个段。 
      * 通过独立的Reduce和Down-sweep CUDA内核实现分段Scan。 
      * **过程**：
        1.  Reduce步骤结束时，将部分和树和部分OR树写入全局内存，保存状态以便稍后开始Down-sweep步骤。 
        2.  执行第二级分段Scan。输入是每个块的部分和树和部分OR树的最后一个元素的标志和数据，以及每个块标志向量的第一个元素。  (https://www.google.com/search?q=%E5%A6%82%E6%9E%9CB%E5%9D%97%EF%BC%8C%E6%AF%8F%E4%B8%AA%E8%BE%93%E5%85%A5%E5%90%91%E9%87%8F%E9%95%BFB%E4%B8%AA%E5%85%83%E7%B4%A0%E3%80%82) 
        3.  第二级分段Scan完成后，运行顶层分段Scan的Down-sweep阶段。 
        4.  从全局内存重新加载Reduce阶段保存的状态。 
        5.  将每个块的最后一个元素赋值为第二级分段Scan输出中对应的元素（而不是像非分段Scan那样赋0）。 
        6.  然后像单块情况一样执行Down-sweep。 

**多块分段Scan算法 (Algorithm 5)** 

```
Algorithm 5: A multi-block segmented scan algorithm.
1: Perform reduce on all blocks in parallel
2: Save partial sum and partial OR trees to global memory
3: Do second-level segmented scans with final sums
4: Load partial sum and partial OR trees from global memory to shared memory
5: Set last element of each block to corresponding element in the output of second-level segmented scan
6: Perform down-sweep on all blocks in parallel
```

### 2.4. 基于Scan构建的原语 (Primitives Built Atop Scan) 

Scan原语被用于实现几种更高级的原语，它们非常适合作为通用任务的构建块。 

  * **2.4.1. Enumerate (枚举)** 

      * **输入**：一个向量和每个元素的真/假值。 
      * **功能**：可以用于分段或非分段输入。 
      * **输出**：对于每个输入元素，输出是该元素左侧真元素的计数。 
      * **实现**：通过将每个真元素设置为1，每个假元素设置为0，然后对该临时向量进行（独占）Scan来实现。 
      * **用途**：在pack（压缩）操作中很有用，只保留标记为真的元素。对于每个真元素，enumerate的输出是必须散射到的地址。 
      * **示例**：`enumerate([t f f t f t t]) = [0 1 1 1 2 2 3]` 

  * **2.4.2. Distribute (复制)** 

      * **功能**：与enumerate类似，可用于分段或非分段输入，并可向前或向后执行。 
      * **输出**：将段的头部（或尾部）元素复制到该段的所有其他元素。 
      * **实现**：
          * **分段输入**：将所有非头部元素设置为0，然后对该向量执行分段包含Scan。 
          * **非分段输入**：一个线程将头部元素写入共享内存，其他线程读取该共享元素会更高效。 
      * **示例**：`distribute([a b c] [d e]) = [a a a] [d d]` 

  * **2.4.3. Split and Split-and-Segment (分割与分段分割)** 

      * **Split (分割)**：
          * **输入**：元素向量和真/假值向量。 
          * **功能**：将输入向量分成两部分，所有标记为假的元素在输出向量的左侧，所有标记为真的元素在右侧。 
          * **实现**：Blelloch使用两次enumerate（需要两次Scan）实现split，一次用于假的元素向前，另一次用于真的元素向后。本文将其减少为一次enumerate，并进行额外的计算来导出真实地址。 
      * **Split-and-segment (分段分割)**：
          * **功能**：作用于分段输入，并对每个段执行稳定的独立分割，此外还将每个段分成两个子段，一个用于假的元素，一个用于真的元素。 
          * **实现**：需要额外的Scan来复制每个段中假的元素数量到该段的所有元素。本文使用类似技术将Blelloch的3次Scan分段分割减少为2次Scan。 
      * **示例**：`split-and-segment([at bf ct] [df et ff]) = [bf] [at ct] [df ff] [et]` 

## 3. 实验方法 

  * **硬件**：NVIDIA GeForce 8800 GTX GPU通过PCI Express 16x连接到Intel Xeon 3.0 GHz CPU。 
  * **软件**：Windows XP，NVIDIA驱动版本97.73，CUDA Toolkit测试版 (2007年2月14日)。 
  * **计时**：仅考虑GPU计算时间，不包括GPU之间的数据传输时间。  为了分摊启动成本，通常串行运行多次计算，并给出平均运行时间。 

### 3.1. G80硬件的影响 

  * NVIDIA 8系列GPU提供了新的硬件功能，为GPGPU工具箱增加了新能力。 
  * **两个重要特性**：
    1.  **共享内存 (Shared Memory)**：每个多处理器增加16 KB的共享内存。  这使得线程之间的数据共享更高效，并通过消除到主GPU内存的流量来提高整体性能。  共享内存没有引入旧硬件上没有的新功能。 
    2.  **散射 (Scatter)**：在GPU可编程单元内支持广义散射。  散射引入了旧NVIDIA GPU上不存在的功能。  虽然Scan和分段Scan原语可以在旧硬件上实现，但散射实现了更高的性能和存储效率。 
          * **散射与收集**：散射对于pack等操作是高效但非必需的；当扫描数据项的向量有序时（如pack使用的enumerate操作），Horn的gather-search操作可以模拟散射，但代价是 $\\log n$ 次遍历。 
          * **AMD GPU**：AMD GPU的R520和R600系列也提供了散射功能，因此本文描述的技术也直接适用于AMD GPU。 

## 4. 应用 

### 4.1. 快速排序 (Quicksort) 

  * **传统挑战**：快速排序因其控制复杂性和不规则并行性，之前难以在GPU上实现。 
  * **基于分段Scan的实现**：分段Scan原语提供了优雅的快速排序公式，归功于Blelloch。 
  * **算法概述**：
      * 算法在输入的所有段上并行运行，元素（线程）之间的所有通信都在单个段内，因此分段Scan原语非常适合。 
      * 在每个段中选择一个枢轴元素（本文选择段的第一个元素），并将其分发到整个段中。 
      * 将输入元素与枢轴进行比较。在算法的交替遍历中，比较大于或大于等于。 
      * 比较产生一个分段的真假向量，用于分割和分段输入；较小的元素移动到向量的头部，较大的元素移动到尾部。每个段分裂成两个。 
      * 从跨越整个输入的单个段开始，当输出排序完成后结束（通过每一步后的全局归约检查）。 
  * **与Blelloch的区别**：Blelloch总是使用相同的比较并使用3向分割，而不是交替比较和2向分割。  作者发现3向分割增加的控制复杂性和所需的额外分段Scan数量不值得其减少的总体遍历次数。 

### 4.2. 稀疏矩阵向量乘法 (Sparse Matrix-Vector Multiply, SpMV) 

  * **GPU与矩阵计算**：基于矩阵的数值计算非常适合GPU，因为它们计算密集且表现出显著的并行性。 
  * **稀疏矩阵的挑战**：稀疏矩阵的吸引力在于比其密集矩阵版本需要更少的存储和计算。理想的稀疏矩阵表示只存储非零元素，不需要填充或排序。  然而，稀疏矩阵的不规则性使得对其操作进行并行化变得困难。 
  * **现有工作**：
      * Bolz等人  的“锯齿对角线”稀疏矩阵表示法  比真正稀疏的表示法更易于并行化，但需要对其行进行排序以使其按长度降序排列。 
      * Krüger和Westermann  对每行中的每四个非零条目渲染一个单独的点。 
      * Brook的spMatrixVec测试使用压缩稀疏行格式，但并行运行在行上，因此每行的运行时间与任何行中最大元素数量成比例。 
  * **本文选择的格式**：作者选择压缩稀疏行 (CSR) 格式，因为它不需要预处理或填充，是数值计算社区中首选的表示之一。 
  * **CSR表示和SpMV算法**：大致遵循Blelloch等人的工作。 
      * **数据结构**：一个 $n \\times n$ 的CSR稀疏矩阵包含 $e$ 个非零元素（条目），由以下三个数据结构表示：
        1.  **value向量**：包含矩阵中所有 $e$ 个非零元素，按Scan顺序（从左到右，从上到下）读取，存储为浮点向量。 
        2.  **index向量**：包含一个整数，标识value向量中每个元素的列。 
        3.  **rowPtr向量**：包含一个整数索引，指向value中每行的第一个元素。 
      * **乘法步骤**：用长度为 $n$ 的向量 `x` 乘以矩阵，并将结果加到长度为 $n$ 的向量 `y`。这些数据结构在GPU主内存中，以及额外的标志和产品临时数据结构（每个包含 $e$ 个条目），矩阵乘法分四步进行： 
        1.  第一个内核遍历所有条目。对于每个条目，将对应的标志设置为0，并对每个条目执行乘法：`product = x[index] * value`。 
        2.  下一个内核遍历所有行，并通过散射将 `flag` 中每个 `rowPtr` 的头部标志设置为1。这为每行创建一个段。 
        3.  然后对 `product` 中 $e$ 个元素以及 `flag` 中的头部标志执行向后分段包含和Scan。 
        4.  最后，运行最终内核遍历所有行，将 `y` 中的值加到从 `products[idx]` 收集的值中。 

## 补充知识点

### 1\. 前缀和（Prefix Sum / Scan）

前缀和（也称为扫描操作）是一种重要的并行计算原语。
给定一个操作 $\\oplus$ 和一个序列 $[a\_0, a\_1, \\dots, a\_{n-1}]$：

  * **独占前缀和 (Exclusive Prefix Sum)**：
    $out\_0 = i$ (identity element)
    $out\_k = a\_0 \\oplus a\_1 \\oplus \\dots \\oplus a\_{k-1}$ for $k \> 0$
    例如，对于加法和初始序列 $[1, 2, 3, 4]$，单位元为0：
    $out = [0, 1, 1+2, 1+2+3] = [0, 1, 3, 6]$

  * **包含前缀和 (Inclusive Prefix Sum)**：
    $out\_k = a\_0 \\oplus a\_1 \\oplus \\dots \\oplus a\_k$
    例如，对于加法和初始序列 $[1, 2, 3, 4]$：
    $out = [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]$

**应用场景**：
前缀和在各种并行算法中都有广泛应用，例如：

  * **数据压缩/过滤 (Pack/Compaction)**：只保留满足特定条件的元素，并将其“打包”到数组的前部。前缀和可以用来计算每个保留元素的新位置。
  * **并行排序**：如论文中提到的快速排序。
  * **流处理**：累积和、滑动窗口计算等。
  * **图像处理**：例如，计算图像的积分图（Summed-Area Table），用于快速计算任意矩形区域内的像素和。
  * **稀疏数据结构操作**：如论文中的稀疏矩阵向量乘法。

### 2\. CUDA编程模型基础

CUDA（Compute Unified Device Architecture）是NVIDIA推出的并行计算平台和编程模型，允许开发者利用GPU的并行处理能力。

  * **Host (主机)**：CPU及其系统内存。
  * **Device (设备)**：GPU及其显存。
  * **Kernel (内核)**：在GPU上执行的C/C++函数，由大量并行线程执行。
  * **Grid (网格)**：一个Kernel的执行实例，包含多个线程块。
  * **Block (线程块)**：由一组可以协作的线程组成，这些线程可以访问共享内存并同步。
  * **Thread (线程)**：最基本的执行单元。
  * **内存层次**：
      * **寄存器 (Registers)**：每个线程私有，速度最快。
      * **共享内存 (Shared Memory)**：同一线程块内的线程共享，片上存储，速度快，由程序员显式管理。
      * **局部内存 (Local Memory)**：每个线程私有，位于显存中，速度较慢。
      * **常量内存 (Constant Memory)**：全局可见，只读，速度快（有缓存）。
      * **纹理内存 (Texture Memory)**：全局可见，只读，有特殊缓存用于2D/3D空间局部性。
      * **全局内存 (Global Memory)**：所有线程可见，位于显存中，速度最慢但容量最大。

**示例代码结构 (CUDA C++)**：

```cpp
// CUDA Kernel function - executed on the GPU
__global__ void addVectors(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code - executed on the CPU
int main() {
    int N = 1024; // Size of vectors
    size_t size = N * sizeof(float);

    // Host vectors
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    // Initialize host vectors (example)
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device vectors
    float* d_a;
    float* d_b;
    float* d_c;

    // Allocate memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    addVectors<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results (example)
    // ...

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

### 3\. Scan的并行算法（例如Bleleloch算法的简化思想）

Scan操作的并行实现通常采用两阶段的树形归约（tree-based reduction）方法，与论文中提到的Reduce和Down-sweep类似。

1.  **上扫 (Up-Sweep / Reduce)**：

      * 每个叶子节点存储输入数组的一个元素。
      * 并行地将相邻节点的值向上累加，直到树的根部。
      * 通常会存储中间节点的值，这些值代表其子树的部分和。
      * 完成此阶段后，树的根节点将包含整个数组的总和（对于包含Scan），或者对于独占Scan，会存储一些需要调整的值。

2.  **下扫 (Down-Sweep / Distribute)**：

      * 从树的根部开始，向下传播信息。
      * 根节点被设置为单位元（对于独占Scan）。
      * 每个父节点将其值传递给左子节点，同时计算并传递一个更新后的值给右子节点，该值包含了左子树的累加和。
      * 通过这种方式，每个叶子节点最终会得到其前面所有元素的累加和。

**数学公式示例 (简化版，对于加法Scan)**：

假设我们有一个数组 $A = [a\_0, a\_1, \\dots, a\_{n-1}]$，要计算独占前缀和 $S = [s\_0, s\_1, \\dots, s\_{n-1}]$。

**上扫阶段**：
在每一步 $d$ (从 $0$ 到 $\\log\_2 N - 1$)，对于所有 $k$ (步长为 $2^{d+1}$ ) 并行执行：
$temp[k + 2^{d+1} - 1] = A[k + 2^{d} - 1] + A[k + 2^{d+1} - 1]$
这个过程会将部分和向上累积，最终 $A[N-1]$ （在处理过程中，它变成了根节点的值）将包含所有元素的总和。

**下扫阶段**：

1.  设置 $A[N-1] = 0$ (对于独占Scan的最后一个元素)。
2.  在每一步 $d$ (从 $\\log\_2 N - 1$ 到 $0$)，对于所有 $k$ (步长为 $2^{d+1}$ ) 并行执行：
    $t = A[k + 2^d - 1]$
    $A[k + 2^d - 1] = A[k + 2^{d+1} - 1]$
    $A[k + 2^{d+1} - 1] = t + A[k + 2^{d+1} - 1]$
    这个阶段将累积的和向下传播，同时确保每个元素得到的是其“前面”元素的和。

上述伪代码是论文中Algorithm 1和Algorithm 2的数学化表达。

### 4\. 稀疏矩阵向量乘法 (SpMV)

稀疏矩阵向量乘法是科学计算中的一个核心操作，通常表示为 $y = A \\cdot x$，其中 $A$ 是一个稀疏矩阵，$x$ 和 $y$ 是向量。

**CSR (Compressed Sparse Row) 格式**：
CSR是一种常用的稀疏矩阵存储格式，非常适合行向量乘法。它使用三个数组来表示矩阵：

1.  `value`：存储所有非零元素的值，按行主序排列。
2.  `col_idx`：存储 `value` 中每个元素的列索引。
3.  `row_ptr`：存储一个指针数组，`row_ptr[i]` 指示第 `i` 行的第一个元素在 `value` 和 `col_idx` 数组中的起始位置。`row_ptr[i+1] - row_ptr[i]` 就是第 `i` 行的非零元素数量。最后一个元素 `row_ptr[num_rows]` 等于非零元素的总数。

**SpMV与CSR的计算过程**：
对于每一行 $i$，计算 $y\_i = \\sum\_{j} A\_{ij} \\cdot x\_j$。由于只存储非零元素，这个求和只针对非零元素进行。
在并行环境中，可以将每一行的计算分配给不同的线程或线程块。Scan原语在这里的作用是，如果多行计算的结果需要合并或累加到某个结构中，它可以提供高效的并行求和机制，特别是在处理行内（或段内）的累积和时。论文中提到的“向后分段包含和Scan”正是为了在每个段（行）内累加乘积。

**代码示例 (SpMV with CSR, 概念性C++伪代码)**：

```cpp
// Assume A is a sparse matrix stored in CSR format
// value: array of non-zero values
// col_idx: array of column indices for each non-zero value
// row_ptr: array of pointers to the start of each row in value/col_idx

// x: input vector
// y: output vector (initialized to zeros or some initial values)
// num_rows: number of rows in the matrix
// num_non_zeros: total number of non-zero elements

void sparseMatrixVectorMultiply(
    const float* value,
    const int* col_idx,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows)
{
    for (int i = 0; i < num_rows; ++i) { // Iterate over each row
        float row_sum = 0.0f;
        // Elements for current row start at row_ptr[i] and end before row_ptr[i+1]
        for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            row_sum += value[j] * x[col_idx[j]];
        }
        y[i] = row_sum; // Or y[i] += row_sum for accumulation
    }
}
```

在GPU上，这个循环的每一行可以由一个独立的线程或线程块并行执行。当行内计算完成后，如果需要对多个行进行某种合并（如论文中提到的Tridiagonal Matrix Solver），Scan原语就可以发挥作用。对于稀疏矩阵向量乘法，Scan主要用于处理每一行内的乘积累加，以及在多块处理时进行跨块的累积。