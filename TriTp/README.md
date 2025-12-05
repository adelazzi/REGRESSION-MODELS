# Comparative Study of Sorting Algorithms: Merge Sort vs Quick Sort

## Overview

This project implements a comprehensive comparative study between **Merge Sort** and **Quick Sort** algorithms, analyzing their performance across different data distributions and array sizes. The study follows a rigorous experimental protocol with statistical analysis.

## 🎯 Objectives

- Implement and compare Merge Sort and Quick Sort algorithms
- Analyze performance across different data distributions
- Measure execution time, comparisons, and swaps
- Compare empirical results with theoretical complexity
- Generate statistical analysis with confidence intervals

## 📊 Experimental Protocol

### Algorithms Tested
1. **Merge Sort** - O(n log n) guaranteed complexity
2. **Quick Sort** with two pivot strategies:
   - Median-of-three pivot selection
   - Random pivot selection

### Data Distributions
- **Random**: Randomly distributed integers
- **Sorted**: Already sorted in ascending order
- **Reverse**: Sorted in descending order
- **Nearly Sorted**: Almost sorted with few random swaps

### Test Parameters
- **Array Sizes**: 1,000 | 5,000 | 10,000 | 50,000 elements
- **Iterations**: 30 runs per configuration for statistical significance
- **Metrics Measured**:
  - Execution time (seconds)
  - Number of comparisons
  - Number of swaps/moves

## 🔧 Implementation Features

### Advanced Quick Sort Implementation
- **Introsort Hybrid**: Falls back to heap sort for deep recursion
- **Median-of-three**: Optimized pivot selection for better performance
- **Random Pivot**: Alternative strategy for comparison
- **Recursion Limit Protection**: Prevents stack overflow

### Statistical Analysis
- Mean and standard deviation calculation
- Error bars in visualizations
- Multiple iterations for reliable measurements
- Complexity analysis against theoretical bounds

### Visualization
- Performance comparison charts
- Error bar plots showing variability
- Distribution-specific analysis
- Logarithmic scaling for clarity

## 🚀 Usage

```python
# Run the complete experimental study
python Tptri.py
```

The script will:
1. Execute all algorithm combinations across all distributions
2. Perform statistical analysis (30 iterations each)
3. Generate comparative visualizations
4. Display complexity analysis results

## 📈 Expected Results

### Theoretical Complexity
- **Merge Sort**: O(n log n) in all cases
- **Quick Sort**: 
  - Best/Average: O(n log n)
  - Worst: O(n²) - mitigated by hybrid approach

### Performance Insights
- Merge Sort shows consistent performance across all distributions
- Quick Sort performance varies significantly with data distribution
- Median-of-three pivot generally outperforms random pivot
- Nearly sorted data favors optimized Quick Sort variants

## 📋 Output

The program generates:
1. **Real-time Progress**: Test execution status with timing
2. **Statistical Tables**: Mean comparisons vs theoretical complexity
3. **Performance Graphs**: 
   - Execution time comparison with error bars
   - Comparison count analysis
4. **Comprehensive Analysis**: Detailed performance breakdown

## 🛠️ Technical Details

### Dependencies
- `matplotlib`: For visualization and plotting
- `numpy`: For numerical operations
- `statistics`: For statistical calculations
- `random`: For data generation and randomization
- `time`: For performance measurement

### System Requirements
- Python 3.6+
- Sufficient memory for large array operations
- Increased recursion limit for deep sorting operations

## 📝 Educational Value

This implementation serves as an educational tool for understanding:
- Divide-and-conquer algorithm design
- Performance analysis methodology
- Statistical experimental design
- Algorithm optimization techniques
- Complexity theory validation through empirical testing

## 🔍 Key Insights

The study demonstrates:
- How algorithm choice depends on data characteristics
- The importance of pivot selection in Quick Sort
- Trade-offs between guaranteed performance (Merge Sort) vs average-case optimization (Quick Sort)
- Statistical methodology for algorithm comparison
