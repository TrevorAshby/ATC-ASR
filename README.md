# ATCASR
Air Traffic Control - Automatic Speech Recognition - CS 674: Project 2
|Date|Time|Accomplished|
|---|---|---
3/16/22 | 11:00AM-12:00PM(noon)=**1hr** | Began README, downloaded initial dataset, began looking for resources about the conformer architecture. Started implementing the <a href="https://paperswithcode.com/method/swish">Swish</a> activation function and the complete linear module.
|**Total**|**1hr**|

## Conformer Architecture
- SpecAug
- Conv Subsampling
- Linear
- Dropout
- *Conformer Block* x N

The *Conformer block* is made up of 3 modules [Convolution Module], [MHSA Module], and [Feed Forward Module]:

### Convolution Module
- Layernorm
- Pointwise Conv
- Glu Activation
- 1D Depthwise Conv
- BatchNorm
- Swish Activation
- Pointwise Conv
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
- Swish Activation
- Dropout
- Linear Layer
- Dropout
- Residual with input


## Helpful Links
- Conformer: https://arxiv.org/pdf/2005.08100v1.pdf
- Efficient Conformer: https://arxiv.org/pdf/2109.01163.pdf
- Dataset: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
