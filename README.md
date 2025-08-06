# FEUP - Parallel and Distributed Computing - 2024/2025
> Curricular Unit: CPD - [Computação Paralela e Distribuída](https://sigarra.up.pt/feup/pt/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=541893)
## 3rd Year - 2nd Semester 1st Project
### Brief description:
This project conducts a comprehensive performance analysis of different matrix multiplication algorithms, evaluating their impact on processor performance using both single-core and multi-core processing approaches. The study leverages the Performance API (PAPI) to collect detailed execution metrics and compare algorithm efficiency across various matrix sizes and parallelization strategies.

We implemented several algorithm variants including the standard row-by-column approach, a memory-optimized line algorithm with improved cache utilization, and block-based algorithms for large matrices. Additionally, we developed parallel versions using OpenMP with both external and internal parallelization strategies. The analysis covers execution time, MFLOPS, cache miss patterns, speedup, and parallel efficiency across matrix sizes ranging from 600×600 to 10240×10240.

Our findings demonstrate that optimal matrix multiplication performance depends heavily on memory access patterns, parallelization strategies, and hardware characteristics. The line algorithm achieved 2-4x faster execution than the standard approach, while external parallelization delivered 4-6x speedup over sequential execution.

I hope you find it useful!

### Developed by:
- Tomás Oliveira - up202208415
- Diogo Ferreira - up202205295
- Álvaro Torres - up202208954

## Objectives

### Single-Core Performance Evaluation
- Implement basic matrix multiplication algorithms in C++ and C#
- Measure execution time for matrix sizes ranging from 600×600 to 10240×10240
- Develop optimized algorithms focusing on memory access patterns
- Implement block-oriented multiplication for cache optimization

### Multi-Core Performance Evaluation
- Create parallel versions using OpenMP
- Analyze performance improvements through parallelization
- Compare different parallelization strategies (external vs internal)
- Evaluate metrics: MFLOPS, speedup, and efficiency

## Algorithms Implemented

### 1. Standard Algorithm (`OnMult`)
- **Description**: Basic row-by-column matrix multiplication (i-j-k loop order)
- **Languages**: C++ and C#
- **Characteristics**: Straightforward implementation with suboptimal memory access patterns

### 2. Line Algorithm (`OnMultLine`)
- **Description**: Memory-optimized approach with i-k-j loop ordering
- **Languages**: C++ and C#
- **Advantages**: Better cache utilization through row-wise access patterns

### 3. Block Algorithm (`OnMultBlock`)
- **Description**: Cache-friendly implementation dividing matrices into smaller blocks
- **Languages**: C++ and C#
- **Block Sizes**: 128, 256, 512
- **Benefits**: Optimized for cache hierarchy performance

### 4. Parallel Implementations
- **External Parallelization** (`OnMultLineExtParallel`): Parallelizes outermost loop
- **Internal Parallelization** (`OnMultLineIntParallel`): Parallelizes innermost loop
- **Technology**: OpenMP with `#pragma omp parallel for`

## Performance Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Execution Time** | Algorithm runtime in seconds | Primary performance indicator |
| **MFLOPS** | Mega Floating Point Operations per Second | Computational efficiency measure |
| **L1/L2 Cache Misses** | Cache miss patterns using PAPI | Memory hierarchy analysis |
| **Speedup** | Parallel vs sequential performance ratio | Parallelization effectiveness |
| **Efficiency** | Resource utilization in parallel execution | Scalability assessment |

## Key Findings

### Algorithm Performance
- **Line Algorithm**: 2-4x faster than standard algorithm across all matrix sizes
- **Block Algorithm**: Most effective for very large matrices (8192×8192+)
- **Optimal Block Size**: 512 for large matrices on tested hardware

### Language Comparison
- **C++**: Consistently outperforms C# by 1.5-5.8x depending on algorithm
- **Performance Gap**: Larger for line algorithm (4.3-5.8x) vs standard (1.5-3.1x)

### Parallel Performance
- **External Parallelization**: 4-6x speedup over sequential execution
- **Internal Parallelization**: Limited improvement (0.5-1.3x speedup)
- **Efficiency**: External approach reaches 70% efficiency vs 15% for internal

### Memory Hierarchy Impact
- **Cache Behavior**: Critical factor for large matrices
- **Line Algorithm**: Dramatically fewer L1 cache misses
- **Block Algorithms**: Better L2 cache utilization

## System Specifications

**Test Environment:**
- **Processor**: Intel Core i7-9700 CPU (8 cores)
- **L1 Cache**: 64KB per core (32KB instructions + 32KB data)
- **L2 Cache**: 256KB per core
- **L3 Cache**: 12MB shared
- **Compiler**: C++ with -O2 optimization flag

## How to Run

### Prerequisites
- C++ compiler with OpenMP support
- C# runtime environment (.NET)
- PAPI library for performance monitoring

### Compilation
```bash
# C++ compilation
g++ -O2 -fopenmp -lpapi matrix_mult.cpp -o matrix_mult

# C# compilation
csc matrix_mult.cs
```

### Execution
```bash
# Run C++ version
./matrix_mult

# Run C# version
mono matrix_mult.exe
```

## Project Structure

```
├── src/
│   ├── cpp/
│   │   ├── matrix_mult.cpp
│   │   ├── parallel_implementations.cpp
│   │   └── performance_metrics.cpp
│   └── csharp/
│       └── matrix_mult.cs
├── results/
│   ├── graphs/
│   │   ├── l1_cache_misses.png
│   │   ├── l2_cache_misses.png
│   │   ├── execution_time_comparison.png
│   │   ├── mflops_comparison.png
│   │   ├── speedup_comparison.png
│   │   ├── parallel_efficiency.png
│   │   └── block_comparison_*.png
│   └── performance_data/
├── docs/
│   └── report.pdf
└── README.md
```

## Results Visualization

The project includes comprehensive performance analysis with graphs showing:
- **L1/L2 Cache Misses**: Memory hierarchy performance across algorithms
- **Execution Time Comparisons**: C++ vs C# performance across matrix sizes
- **MFLOPS Performance**: Computational efficiency for different implementations
- **Parallel Speedup**: External vs internal parallelization effectiveness
- **Block Size Optimization**: Performance impact of different block sizes
- **Parallel Efficiency**: Resource utilization across matrix sizes

## Algorithm Details

### Standard Matrix Multiplication (OnMult)
```cpp
for (int i = 0; i < m_ar; i++) {
    for (int j = 0; j < m_br; j++) {
        double temp = 0;
        for (int k = 0; k < m_ar; k++) {
            temp += matrixA[i * m_ar + k] * matrixB[k * m_br + j];
        }
        matrixC[i * m_ar + j] = temp;
    }
}
```

### Line-Oriented Algorithm (OnMultLine)
```cpp
for (int i = 0; i < m_ar; i++) {
    for (int k = 0; k < m_ar; k++) {
        double temp = matrixA[i * m_ar + k];
        for (int j = 0; j < m_br; j++) {
            matrixC[i * m_ar + j] += temp * matrixB[k * m_br + j];
        }
    }
}
```

### External Parallelization
```cpp
#pragma omp parallel for
for (int i = 0; i < m_ar; i++) {
    for (int k = 0; k < m_ar; k++) {
        double temp = matrixA[i * m_ar + k];
        for (int j = 0; j < m_br; j++) {
            matrixC[i * m_ar + j] += temp * matrixB[k * m_br + j];
        }
    }
}
```

## Technologies Used

- **Languages**: C++, C#
- **Parallelization**: OpenMP
- **Performance Monitoring**: PAPI (Performance Application Programming Interface)
- **Timing**: OpenMP high-precision timers (`omp_get_wtime()`), C# Stopwatch
- **Optimization**: Compiler optimizations (-O2), cache-aware algorithms
- **Memory Management**: Dynamic allocation with manual cleanup

## Performance Summary

### Matrix Size Range Tested
- **Small**: 600×600 to 3000×3000 (increments of 400)
- **Large**: 4096×4096 to 10240×10240 (increments of 2048)

### Best Performing Configurations
1. **Single-threaded**: Line algorithm in C++
2. **Multi-threaded**: External parallelization of line algorithm
3. **Large matrices**: Block algorithm with 512 block size
4. **Cache efficiency**: Line algorithm with minimal L1/L2 misses

## Conclusions

The study demonstrates that optimal matrix multiplication performance requires:

1. **Memory-aware algorithms** (line algorithm) for cache efficiency
2. **External parallelization strategies** to minimize synchronization overhead  
3. **Block-based approaches** for very large matrices
4. **Language considerations** where C++ provides significant performance advantages
5. **Hardware awareness** of cache hierarchy and memory bandwidth limitations

The research provides valuable insights into the relationship between algorithm design, memory hierarchy, and parallel processing for computational linear algebra operations.

## Additional Resources

- Full project report with detailed analysis and graphs
- Performance measurement data and raw results
- Complete source code with detailed comments
- Comparative analysis across different hardware configurations

---

*This project was developed as part of the Parallel and Distributed Computing course at FEUP, demonstrating practical applications of performance optimization techniques in high-performance computing.*
