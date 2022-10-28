### Kernel Metric Learning for Importance Sampling (KMIS)

This repository contains an implementation of KMIS in the paper: "Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions" by Haanvid Lee, Jongmin Lee, Yunseon Choi, Wonseok Jeon, Byung-Jun Lee, Yung-Kyun Noh, and Kee-Eung Kim. It was published as a conference paper at the Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS) 2022.

The codes are used for conducting the experiments on the absolute error domain in the "Experiments" Section. Using the codes, data required for the experiments can be generated, and the target policy described in the paper can be evaluated with the generated data. Our codes are written based on the codes of Kallus and Zhou [1].

For the implementation of the work of Kallus and Zhou [1] and SLOPE [2], codes written by the authors at  https://github.com/CausalML/continuous-policy-learning  and  https://github.com/VowpalWabbit/slope-experiments were adapted, respectively.

### Conda Environment Installation

`$ conda env create -f environment.yml`

### How to Run

`$ python main_dummy.py`

### Bibtex
If you use this code, please cite our paper:

```
@inproceedings{lee2022local,
title={Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions},
author={Haanvid Lee and Jongmin Lee and Yunseon Choi and Wonseok Jeon and Byung-Jun Lee and Yung-Kyun Noh and Kee-Eung Kim},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022}
}
```

### References
[1] Kallus, Nathan, and Angela Zhou. "Policy evaluation and optimization with continuous treatments." International Conference on Artificial Intelligence and Statistics. PMLR, 2018.

[2] Su, Yi, Pavithra Srinath, and Akshay Krishnamurthy. "Adaptive estimator selection for off-policy evaluation." International Conference on Machine Learning. PMLR, 2020.