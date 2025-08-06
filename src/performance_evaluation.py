import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    cs_path = "metrics_cs/results_cs.csv"
    cpp_path = "metrics_cpp/results_cpp.csv"
    
    if not os.path.exists(cs_path):
        print(f"Warning: {cs_path} not found!")
        cs_data = None
    else:
        cs_data = pd.read_csv(cs_path)
    
    if not os.path.exists(cpp_path):
        print(f"Warning: {cpp_path} not found!")
        cpp_data = None
    else:
        cpp_data = pd.read_csv(cpp_path)
    
    return cs_data, cpp_data

def plot_execution_time_comparison(cs_data, cpp_data, save_path="plots"):
    plt.figure(figsize=(12, 8))
    
    sizes = [600, 1000, 1400, 1800, 2200, 2600, 3000]
   
    cs_std = cs_data[cs_data['algorithm'] == 'Standard']
    cs_line = cs_data[cs_data['algorithm'] == 'Line']
    
    cpp_std = cpp_data[cpp_data['algorithm'] == 'Standard']
    cpp_line = cpp_data[cpp_data['algorithm'] == 'Line']
    
    plt.plot(cs_std['size'], cs_std['time'], 'o-', label='C# Standard')
    plt.plot(cs_line['size'], cs_line['time'], 's-', label='C# Line')
    plt.plot(cpp_std['size'], cpp_std['time'], 'o--', label='C++ Standard')
    plt.plot(cpp_line['size'], cpp_line['time'], 's--', label='C++ Line')
    
    plt.title('Execution Time Comparison: C# vs C++')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "time_comparison_standard_line.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/time_comparison_standard_line.png")

def plot_block_size_comparison(cs_data, cpp_data, save_path="plots"):
    plt.figure(figsize=(12, 8))
    
    large_sizes = [4096, 6144, 8192, 10240]
    block_sizes = [128, 256, 512]
    
    for size in large_sizes:
        plt.figure(figsize=(10, 6))
        
        for bs in block_sizes:
            cs_block = cs_data[cs_data['algorithm'] == f'Block_{bs}']
            cs_block = cs_block[cs_block['size'] == size]
            
            cpp_block = cpp_data[cpp_data['blockSize'] == bs]
            cpp_block = cpp_block[cpp_block['size'] == size]
            
            if not cs_block.empty:
                plt.bar(f'C# Block {bs}', cs_block['time'].values[0], label=f'C# Block {bs}')
            
            if not cpp_block.empty:
                plt.bar(f'C++ Block {bs}', cpp_block['time'].values[0], label=f'C++ Block {bs}')
        
        plt.title(f'Block Multiplication Performance - Size {size}x{size}')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y')
        plt.tight_layout()
 
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"block_comparison_{size}.png"))
        plt.close()
        
        print(f"Plot saved to {save_path}/block_comparison_{size}.png")

def plot_speedup_comparison(cpp_data, save_path="plots"):
    """Speedup and efficiency for C++ parallel implementations"""
    if 'speedup' not in cpp_data.columns:
        print("Speedup data not available in C++ results.")
        return
    
    plt.figure(figsize=(12, 8))
  
    ext_parallel = cpp_data[cpp_data['algorithm'].str.contains('ExtParallel')]
    int_parallel = cpp_data[cpp_data['algorithm'].str.contains('IntParallel')]
 
    plt.plot(ext_parallel['size'], ext_parallel['speedup'], 'o-', label='External Parallel')
    plt.plot(int_parallel['size'], int_parallel['speedup'], 's-', label='Internal Parallel')
    
    plt.title('Speedup Comparison for Parallel C++ Implementations')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
 
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "speedup_comparison.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/speedup_comparison.png")

def plot_relative_performance(cs_data, cpp_data, save_path="plots"):
    """Relative performance between C# and C++ (C++ as baseline)"""
    plt.figure(figsize=(12, 8))

    cs_sizes = set(cs_data['size'].unique())
    cpp_sizes = set(cpp_data['size'].unique())
    common_sizes = sorted(list(cs_sizes.intersection(cpp_sizes)))
    
    cpp_to_cs_mapping = {
        'Standard': 'Standard',
        'Line': 'Line',
        'Line_large': 'Line',
    }
 
    labels = []
    std_ratios = []
    line_ratios = []
    block_128_ratios = []
    block_256_ratios = []
    block_512_ratios = []
    
    for size in common_sizes:
        labels.append(str(size))

        cs_std = cs_data[(cs_data['algorithm'] == 'Standard') & (cs_data['size'] == size)]
       
        cpp_std = cpp_data[(cpp_data['algorithm'] == 'Standard') & (cpp_data['size'] == size)]
       
        cs_line = cs_data[(cs_data['algorithm'] == 'Line') & (cs_data['size'] == size)]
        cpp_line = cpp_data[((cpp_data['algorithm'] == 'Line') | (cpp_data['algorithm'] == 'Line_large')) & 
                           (cpp_data['size'] == size)]
        
        cs_block_128 = cs_data[(cs_data['algorithm'] == 'Block_128') & (cs_data['size'] == size)]
        cpp_block_128 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & 
                               (cpp_data['blockSize'] == 128) & 
                               (cpp_data['size'] == size)]
        
        cs_block_256 = cs_data[(cs_data['algorithm'] == 'Block_256') & (cs_data['size'] == size)]
        cpp_block_256 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & 
                               (cpp_data['blockSize'] == 256) & 
                               (cpp_data['size'] == size)]
        
        cs_block_512 = cs_data[(cs_data['algorithm'] == 'Block_512') & (cs_data['size'] == size)]
        cpp_block_512 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & 
                               (cpp_data['blockSize'] == 512) & 
                               (cpp_data['size'] == size)]
        
        try:
            if not cs_std.empty and not cpp_std.empty:
                std_ratios.append(cs_std['time'].values[0] / cpp_std['time'].values[0])
            else:
                std_ratios.append(None)
        except (IndexError, KeyError):
            std_ratios.append(None)
            
        try:
            if not cs_line.empty and not cpp_line.empty:
                line_ratios.append(cs_line['time'].values[0] / cpp_line['time'].values[0])
            else:
                line_ratios.append(None)
        except (IndexError, KeyError):
            line_ratios.append(None)
            
        try:
            if not cs_block_128.empty and not cpp_block_128.empty:
                block_128_ratios.append(cs_block_128['time'].values[0] / cpp_block_128['time'].values[0])
            else:
                block_128_ratios.append(None)
        except (IndexError, KeyError):
            block_128_ratios.append(None)
            
        try:
            if not cs_block_256.empty and not cpp_block_256.empty:
                block_256_ratios.append(cs_block_256['time'].values[0] / cpp_block_256['time'].values[0])
            else:
                block_256_ratios.append(None)
        except (IndexError, KeyError):
            block_256_ratios.append(None)
            
        try:
            if not cs_block_512.empty and not cpp_block_512.empty:
                block_512_ratios.append(cs_block_512['time'].values[0] / cpp_block_512['time'].values[0])
            else:
                block_512_ratios.append(None)
        except (IndexError, KeyError):
            block_512_ratios.append(None)

    x_pos = np.arange(len(labels))
    bar_width = 0.15

    def safe_plot_bar(position, values, label, color=None):
        valid_positions = []
        valid_values = []
        for i, val in enumerate(values):
            if val is not None:
                valid_positions.append(position[i])
                valid_values.append(val)
        if valid_values:
            plt.bar(valid_positions, valid_values, width=bar_width, label=label, color=color)

    safe_plot_bar(x_pos - 2*bar_width, std_ratios, 'Standard')
    safe_plot_bar(x_pos - bar_width, line_ratios, 'Line')
    safe_plot_bar(x_pos, block_128_ratios, 'Block 128')
    safe_plot_bar(x_pos + bar_width, block_256_ratios, 'Block 256')
    safe_plot_bar(x_pos + 2*bar_width, block_512_ratios, 'Block 512')
    
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='C++ Baseline')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('C# Time / C++ Time (Higher = C# Slower)')
    plt.title('Relative Performance: C# vs C++')
    plt.xticks(x_pos, labels)
    plt.legend()
    plt.grid(axis='y')

    for i, ratio in enumerate(std_ratios):
        if ratio is not None:
            plt.text(x_pos[i] - 2*bar_width, ratio + 0.1, f"{ratio:.1f}x", 
                     ha='center', va='bottom', fontsize=8)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "relative_performance.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/relative_performance.png")
    
    comparison_data = {
        'Matrix Size': labels,
        'Standard Ratio (C#/C++)': std_ratios,
        'Line Ratio (C#/C++)': line_ratios,
        'Block_128 Ratio (C#/C++)': block_128_ratios,
        'Block_256 Ratio (C#/C++)': block_256_ratios, 
        'Block_512 Ratio (C#/C++)': block_512_ratios
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(save_path, "performance_comparison.csv"), index=False)
    print(f"Detailed comparison saved to {save_path}/performance_comparison.csv")

def plot_mflops_comparison(cpp_data, save_path="plots"):
    """MFLOPS for different implementations in C++"""
    if 'mflops' not in cpp_data.columns:
        print("MFLOPS data not available in C++ results.")
        return
        
    plt.figure(figsize=(12, 8))
    
    std_data = cpp_data[cpp_data['algorithm'] == 'Standard']
    line_data = cpp_data[cpp_data['algorithm'] == 'Line']
    line_large_data = cpp_data[cpp_data['algorithm'] == 'Line_large']
    
    block_128 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 128)]
    block_256 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 256)]
    block_512 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 512)]
    
    plt.plot(std_data['size'], std_data['mflops'], 'o-', label='Standard')
    plt.plot(line_data['size'], line_data['mflops'], 's-', label='Line')
    if not line_large_data.empty:
        plt.plot(line_large_data['size'], line_large_data['mflops'], '^-', label='Line (Large)')
    
    plt.plot(block_128['size'], block_128['mflops'], 'x-', label='Block 128')
    plt.plot(block_256['size'], block_256['mflops'], '*-', label='Block 256')
    plt.plot(block_512['size'], block_512['mflops'], 'd-', label='Block 512')
    
    plt.title('MFLOPS Comparison for Different Algorithms')
    plt.xlabel('Matrix Size')
    plt.ylabel('MFLOPS')
    plt.grid(True)
    plt.legend()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "mflops_comparison.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/mflops_comparison.png")

    plot_parallel_mflops(cpp_data, save_path)

def plot_cache_misses(cpp_data, save_path="plots"):
    """L1 and L2 cache misses for different algorithms"""
    if 'L1' not in cpp_data.columns or 'L2' not in cpp_data.columns:
        print("Cache miss data (L1/L2) not available in C++ results.")
        return

    plt.figure(figsize=(12, 8))
    
    std_data = cpp_data[cpp_data['algorithm'] == 'Standard']
    line_data = cpp_data[cpp_data['algorithm'] == 'Line']
    line_large_data = cpp_data[cpp_data['algorithm'] == 'Line_large']
 
    block_128 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 128)]
    block_256 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 256)]
    block_512 = cpp_data[(cpp_data['algorithm'].str.contains('Block')) & (cpp_data['blockSize'] == 512)]
 
    plt.plot(std_data['size'], std_data['L1']/1e6, 'o-', label='Standard')
    plt.plot(line_data['size'], line_data['L1']/1e6, 's-', label='Line')
    if not line_large_data.empty:
        plt.plot(line_large_data['size'], line_large_data['L1']/1e6, '^-', label='Line (Large)')
    
    plt.plot(block_128['size'], block_128['L1']/1e6, 'x-', label='Block 128')
    plt.plot(block_256['size'], block_256['L1']/1e6, '*-', label='Block 256')
    plt.plot(block_512['size'], block_512['L1']/1e6, 'd-', label='Block 512')
    
    plt.title('L1 Cache Misses for Different Algorithms')
    plt.xlabel('Matrix Size')
    plt.ylabel('L1 Cache Misses (millions)')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "l1_cache_misses.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/l1_cache_misses.png")

    plt.figure(figsize=(12, 8))
    
    plt.plot(std_data['size'], std_data['L2']/1e6, 'o-', label='Standard')
    plt.plot(line_data['size'], line_data['L2']/1e6, 's-', label='Line')
    if not line_large_data.empty:
        plt.plot(line_large_data['size'], line_large_data['L2']/1e6, '^-', label='Line (Large)')
    
    plt.plot(block_128['size'], block_128['L2']/1e6, 'x-', label='Block 128')
    plt.plot(block_256['size'], block_256['L2']/1e6, '*-', label='Block 256')
    plt.plot(block_512['size'], block_512['L2']/1e6, 'd-', label='Block 512')
    
    plt.title('L2 Data Cache Misses (DCM) for Different Algorithms')
    plt.xlabel('Matrix Size')
    plt.ylabel('L2 Cache Misses (millions)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, "l2_cache_misses.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/l2_cache_misses.png")
    
    plot_parallel_cache_misses(cpp_data, save_path)

def plot_parallel_mflops(cpp_data, save_path="plots"):
    """MFLOPS for parallel implementations"""
    if 'mflops' not in cpp_data.columns:
        return
        
    plt.figure(figsize=(12, 8))

    ext_parallel = cpp_data[cpp_data['algorithm'].str.contains('ExtParallel')]
    int_parallel = cpp_data[cpp_data['algorithm'].str.contains('IntParallel')]
    line_serial = cpp_data[cpp_data['algorithm'] == 'Line']

    if not line_serial.empty:
        plt.plot(line_serial['size'], line_serial['mflops'], 'o--', label='Serial Line')
    
    plt.plot(ext_parallel['size'], ext_parallel['mflops'], 's-', label='External Parallel')
    plt.plot(int_parallel['size'], int_parallel['mflops'], 'x-', label='Internal Parallel')
    
    plt.title('MFLOPS Comparison for Parallel Implementations')
    plt.xlabel('Matrix Size')
    plt.ylabel('MFLOPS')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "parallel_mflops.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/parallel_mflops.png")

def plot_parallel_cache_misses(cpp_data, save_path="plots"):
    """Cache misses for parallel implementations"""
    if 'L1' not in cpp_data.columns or 'L2' not in cpp_data.columns:
        return

    plt.figure(figsize=(12, 8))
 
    ext_parallel = cpp_data[cpp_data['algorithm'].str.contains('ExtParallel')]
    int_parallel = cpp_data[cpp_data['algorithm'].str.contains('IntParallel')]
    line_serial = cpp_data[cpp_data['algorithm'] == 'Line']

    if not line_serial.empty:
        plt.plot(line_serial['size'], line_serial['L1']/1e6, 'o--', label='Serial Line') 
    
    plt.plot(ext_parallel['size'], ext_parallel['L1']/1e6, 's-', label='External Parallel')
    plt.plot(int_parallel['size'], int_parallel['L1']/1e6, 'x-', label='Internal Parallel')
    
    plt.title('L1 Cache Misses for Parallel Implementations')
    plt.xlabel('Matrix Size')
    plt.ylabel('L1 Cache Misses (millions)')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "parallel_l1_misses.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/parallel_l1_misses.png")

    plt.figure(figsize=(12, 8))

    if not line_serial.empty:
        plt.plot(line_serial['size'], line_serial['L2']/1e6, 'o--', label='Serial Line')
    
    plt.plot(ext_parallel['size'], ext_parallel['L2']/1e6, 's-', label='External Parallel')
    plt.plot(int_parallel['size'], int_parallel['L2']/1e6, 'x-', label='Internal Parallel')
    
    plt.title('L2 Cache Misses for Parallel Implementations')
    plt.xlabel('Matrix Size')
    plt.ylabel('L2 Cache Misses (millions)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, "parallel_l2_misses.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/parallel_l2_misses.png")

def plot_parallel_efficiency(cpp_data, save_path="plots"):
    """Efficiency for parallelization"""
    if 'efficiency' not in cpp_data.columns:
        print("Efficiency data not available in C++ results.")
        return
        
    plt.figure(figsize=(12, 8))

    ext_parallel = cpp_data[cpp_data['algorithm'].str.contains('ExtParallel')]
    int_parallel = cpp_data[cpp_data['algorithm'].str.contains('IntParallel')]
    
    plt.plot(ext_parallel['size'], ext_parallel['efficiency'], 'o-', label='External Parallel')
    plt.plot(int_parallel['size'], int_parallel['efficiency'], 's-', label='Internal Parallel')

    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Perfect Efficiency')
    
    plt.title('Efficiency of Parallel Implementations')
    plt.xlabel('Matrix Size')
    plt.ylabel('Efficiency (0-1)')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "parallel_efficiency.png"))
    plt.close()
    
    print(f"Plot saved to {save_path}/parallel_efficiency.png")

def custom_plot_menu(cs_data, cpp_data):
    save_path = input("Enter directory to save plots (default: 'plots'): ") or "plots"
    
    while True:
        print("\nCustom Plot Menu:")
        print("1. Time Comparison: C# vs C++ for Standard & Line algorithms")
        print("2. Block Size Comparison (128, 256, 512) for large matrices")
        print("3. Speedup Comparison for C++ Parallel Implementations")
        print("4. Relative Performance (C# time / C++ time)")
        print("5. MFLOPS Comparison")
        print("6. Cache Misses (L1/L2) Comparison")
        print("7. Parallel Efficiency")
        print("8. Generate All Basic Plots")
        print("9. Generate All Advanced Plots (MFLOPS, Cache misses)")
        print("10. Generate All Plots")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            plot_execution_time_comparison(cs_data, cpp_data, save_path)
        elif choice == '2':
            plot_block_size_comparison(cs_data, cpp_data, save_path)
        elif choice == '3':
            plot_speedup_comparison(cpp_data, save_path)
        elif choice == '4':
            plot_relative_performance(cs_data, cpp_data, save_path)
        elif choice == '5':
            plot_mflops_comparison(cpp_data, save_path)
        elif choice == '6':
            plot_cache_misses(cpp_data, save_path)
        elif choice == '7':
            plot_parallel_efficiency(cpp_data, save_path)
        elif choice == '8':
            plot_execution_time_comparison(cs_data, cpp_data, save_path)
            plot_block_size_comparison(cs_data, cpp_data, save_path)
            plot_speedup_comparison(cpp_data, save_path)
            plot_relative_performance(cs_data, cpp_data, save_path)
        elif choice == '9':
            plot_mflops_comparison(cpp_data, save_path)
            plot_cache_misses(cpp_data, save_path)
            plot_parallel_efficiency(cpp_data, save_path)
        elif choice == '10':
            plot_execution_time_comparison(cs_data, cpp_data, save_path)
            plot_block_size_comparison(cs_data, cpp_data, save_path)
            plot_speedup_comparison(cpp_data, save_path)
            plot_relative_performance(cs_data, cpp_data, save_path)
            plot_mflops_comparison(cpp_data, save_path)
            plot_cache_misses(cpp_data, save_path)
            plot_parallel_efficiency(cpp_data, save_path)
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    print("Matrix Multiplication Performance Visualization Tool")
    print("--------------------------------------------------")
    
    cs_data, cpp_data = load_data()
    
    if cs_data is None and cpp_data is None:
        print("Error: No data files found.")
        return
        
    print(f"Loaded C# data: {cs_data.shape[0] if cs_data is not None else 0} rows")
    print(f"Loaded C++ data: {cpp_data.shape[0] if cpp_data is not None else 0} rows")
    
    custom_plot_menu(cs_data, cpp_data)

if __name__ == "__main__":
    main()