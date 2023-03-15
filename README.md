# Bridging Vision and Language Encoders: Parameter-Efficient Tuning for Referring Image Segmentation

This is an official PyTorch implementation of the ETRIS.


<div align="center">
<img src="img/demo.gif" alt="teaser" style="zoom:50%;" />
</div>
## Overall Architecture

<img src="img/arch.png">


## Preparation

1. Environment
   - [PyTorch](www.pytorch.org) (e.g. 1.10.0)
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)


## Quick Start

To do training of ETRIS, modify the script according to your requirement and run:

```
./run_scripts/train.sh
```

To do evaluation of ETRIS, modify the script according to your requirement and run:

```
./run_scripts/test.sh
```
