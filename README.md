# FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning

Official code repository of  the paper "FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning" in the proceedings of International Joint Conference on Artificial Intelligence (IJCAI) 2024.


[![arXiv](https://img.shields.io/badge/arXiv-2404.14061-b31b1b.svg)](https://arxiv.org/abs/2404.14061)
 

**Requirements**

Hardware environment: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 18.04.6, Python 3.9, PyTorch 1.11.0 and CUDA 11.8.

Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;

**Training**

Here we take Cora-Louvain-10 Clients as an example:

```python
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain
```


**Cite Us**
Please consider citing our paper if you use this code in your work:
```
@misc{zhu2024fedtad,
      title={FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning}, 
      author={Yinlin Zhu and Xunkai Li and Zhengyu Wu and Di Wu and Miao Hu and Rong-Hua Li},
      year={2024},
      eprint={2404.14061},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```