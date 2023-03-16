# ATCASR
Air Traffic Control - Automatic Speech Recognition - CS 674: Project 2
|Date|Time|Accomplished|
|---|---|---
3/16/22 | 11:00AM-12:30PM=**1.5hrs** | Began README, downloaded initial dataset, began looking for resources about the conformer architecture. Started implementing the <a href="https://paperswithcode.com/method/swish">Swish</a> activation function and the complete linear module.
|**Total**|**1.5hrs**|

## Conformer Architecture
- SpecAug
- Conv Subsampling
- Linear
- Dropout
- *Conformer Block* x N

The *Conformer block* is made up of 3 modules [Convolution Module](#convolution-module), [MHSA Module](#multi-headed-self-attention-module), and [Feed Forward Module](#feed-forward-module):

### Convolution Module
- Layernorm

#### Pointwise Conv
```
Pointwise convolution is convolution with a kernel size of 1x1. This preserves the height and width of the "image", but compresses the number of channels, into a form of encoding representing the 3 channels. 
```
<img src="./pointwise.png">

- Glu Activation
- 1D Depthwise Conv
- BatchNorm

#### Swish Activation: https://paperswithcode.com/method/swish
```
"Swish is an activation function, f(x) = x • sigmoid(ßx), where ß a learnable parameter. Nearly all implementations do not use the learnable parameter ß, in which case the activation function is xσ(x) ("Swish-1").


The function  is exactly the SiLU, which was introduced by other authors before the swish."
```    

- [Pointwise Conv](#pointwise-conv)
- Dropout
- Residual with input

### Multi-Headed Self-Attention Module
- Layernorm
- Multi-Head Self-Attention with Relative Positional Embedding
- Dropout
- residual with input

### Feed Forward Module
- Layernorm
- Linear Layer
- [Swish Activation](#swish-activation-httpspaperswithcodecommethodswish)
- Dropout
- Linear Layer
- Dropout
- Residual with input


## Helpful Links
- Conformer: https://arxiv.org/pdf/2005.08100v1.pdf
- Efficient Conformer: https://arxiv.org/pdf/2109.01163.pdf
- Dataset: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html

## References
```bibtex
@misc{https://doi.org/10.48550/arxiv.2005.08100,
  doi = {10.48550/ARXIV.2005.08100},
  url = {https://arxiv.org/abs/2005.08100},
  author = {Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and Pang, Ruoming},
  keywords = {Audio and Speech Processing (eess.AS), Machine Learning (cs.LG), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Conformer: Convolution-augmented Transformer for Speech Recognition},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@inproceedings{hofbauer2008atcosim,
  title={The ATCOSIM Corpus of Non-Prompted Clean Air Traffic Control Speech.},
  author={Hofbauer, Konrad and Petrik, Stefan and Hering, Horst},
  booktitle={LREC},
  year={2008},
  organization={Citeseer}
}
```
