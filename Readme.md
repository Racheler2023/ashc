# Average  Sensitivity of Hierarchical   $k$\-median Clustering

```markdown
A library for computing the average sensitivity of hierarchical $k$-median clustering and other hierarchical clustering algorithms.

## Example usage
python aver_sen_para.py
python special_example_1.py
```

- `rhst.py`: Implements the RHST algorithm introduced in the ICML 2025 paper.
- `cost_mul_k.py`: Compares the computational cost of various algorithms in $k$-median clustering.
- `greedy.py`: Contains the CLINSS (Clustered Greedy $k$-Median) algorithm.
- `exponential.py`: Implements our proposed exponential algorithm.
- `real_aver_sen.py`: Compares the average sensitivity of our algorithm with that of other algorithms.
- `aver_sen_para.py`: Parallelized version of `real_aver_sen.py`, significantly faster.
- `sensitivity.py`: Calculates the cluster symmetric difference, our measure for average sensitivity.
- `single_linkage_well_cluster_test.py`: Validates that the real dataset has a good clustering structure (appendix).
- Experimental scripts on synthetic datasets:
  - `special_example_1.py`: Focuses on single-linkage clustering.
  - `special_example_2.py`: Implements the CLINSS algorithm.
  - `sys_aver_sen.py`: Evaluates performance under different $\epsilon$ values.

## Reference

- Shijie Li, Weiqiang He, Ruobing Bai, Pan Peng. *Average Sensitivity of Hierarchical $k$-Median Clustering*. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025).

