- `rhst.py` implements the RHST introduced in the paper.
- `cost_mul_k.py` compares the computational cost of various algorithms in k-median clustering.
- `greedy.py` contains the CLINSS (Clustered Greedy k-Median) algorithm.
- `exponential.py` is the implementation of our proposed algorithm.

- `real_aver_sen.py` compares the average sensitivity of our algorithm with that of other algorithms.
- `aver_sen_para.py` is a parallelized version of `real_aver_sen.py` and is much faster.
- `sensitivity.py` contains our method for calculating the cluster symmetric difference.
- `single_linkage_well_cluster_test.py` is the code in the appendix for detecting that the real data set has a good clustering structure
- The three experimental scripts on synthetic datasets are:
  - `special_example_1.py`: Focuses on the single-linkage clustering method.
  - `special_example_2.py`: Implements the CLINSS algorithm.
  - `sys_aver_sen.py`: Evaluates the performance of our algorithm under different \(\epsilon\) settings

