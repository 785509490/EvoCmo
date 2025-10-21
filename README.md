# Fully Tensorized GPU-Accelerated Multi-Population Evolutionary Algorithm for Constrained Multi-objective Optimization

## 1. Introduction

This repository contains the source code for the algorithm presented in the paper:

*Fully Tensorized GPU-accelerated Multi-population Evolutionary Algorithm for Constrained Multiobjective Optimization Problems*

[[arXiv:2509.19821](https://arxiv.org/abs/2509.19821)]

This paper introduces a novel evolutionary algorithm designed to leverage the power of GPUs for solving constrained multi-objective optimization problems efficiently. 

## 2. Comparative CMOEA Algorithms and Test Problems

This implementation includes the proposed algorithm, GMPEA2, and a set of established benchmark problems and comparative algorithms tailored for GPU environments:

*   **Comparative CMOEA Algorithms:**  The repository includes GPU-optimized versions of several state-of-the-art Constrained Multi-objective Evolutionary Algorithms (CMOEAs) for performance comparison. (CCMO, CMOEA-MS, PPS, NSGA-II, EMCMO)

*   **Test Problems:** We provide the following benchmark test suites:
    *   C-DTLZ Test Suite
    *   DC-DTLZ Test Suite
    *   LIR-CMOP Test Suite

These test suites represent a diverse range of constrained multi-objective optimization challenges, allowing for a comprehensive evaluation of algorithm performance.

## 3. How to Run the Algorithm 

This codebase is built as an extension of the [EvoX](https://github.com/EMI-Group/evox) 1.1.2 framework. To set up the environment, please refer to the EvoX README for detailed instructions.

To run a test demo of our algorithm, follow these steps:

1.  **Environment Setup:** Ensure you have the EvoX environment configured correctly, including the necessary GPU drivers and CUDA toolkit.
2.  **Navigate to the test file:** Locate the `test.py` file in the `evox\unit_test\` directory.
3.  **Run the test:** Execute the `test.py` file.  The default test configuration runs the GMPEA2 algorithm on the LIRCMOP9 problem.

    ```bash  
    python evox\unit_test\test.py  

You can modify the test.py file to experiment with different algorithms and test problems. 


## 4. Acknowledgement
Thank you for your interest in our work. We hope this codebase proves valuable for your research and applications in constrained multi-objective optimization.

Please cite the following paper if you use this code in your research:

@article{huang2025fully,
title={Fully Tensorized GPU-accelerated Multi-population Evolutionary Algorithm for Constrained Multiobjective Optimization Problems},
author={Huang, Weixiong and Wang, Rui and Li, Wenhua and Qi, Sheng and Luo, Tianyu and Chen, Delong and Zhang, Tao and Wang, Ling},
journal={arXiv preprint arXiv:2509.19821},
year={2025}
}

This work was supported by the Scientific Research Project of Xiang Jiang Lab (22XJ02003), the University Fundamental Research Fund (23-ZZCX-JDZ-28), the National Science Fund for Outstanding Young Scholars (62122093), the National Natural Science Foundation of China (72071205), the National University of Defense Technology Youth Innovation Science Fund Project (ZK25-62), the science and technology innovation Program of Hunan Province (ZC23112101-10), and the Hunan Natural Science Foundation Regional Joint Project (2023JJ50490). 

The authors would like to thank the support by the COSTA: complex system optimization team of the College of System Engineering at NUDT.

College of Systems Engineering, National University of Defense Technology, Changsha 410073, China (e-mail: huangweixiong@nudt.edu.cn)
