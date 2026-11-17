# BiOPAL: $\textcolor{blue}{\textsf{Bi}}\textsf{level }\textcolor{blue}{\textsf{OP}}\textsf{timizer with an }\textcolor{blue}{\textsf{A}}\textsf{gent in the }\textcolor{blue}{\textsf{L}}\textsf{oop}$

Bilevel Optimizer with an Agent in the Loop for Sequence Prediction. 
Based on OPAL in [Reactive Optimizers](https://arxiv.org/abs/2610.33826) ([Bai et al., 2026](#references)).
 
## Installation
 
Click a system to expand its instructions.
 
<details>
<summary><b>lambda13</b></summary>

```bash
# create conda env
conda create -n biopal python=3.11 pip -y
python -m pip install --upgrade pip setuptools wheel
 
# install deps
python -m pip install \
    "transformers==5.14.1" \
    "accelerate==1.14.0" \
    "numpy==2.3.5"
 
# w/o sudo rights, install in conda env
conda install -c conda-forge bubblewrap socat -y
 
python -m pip install pandas pyarrow
 
# NTv2
python -m pip install "transformers==4.44.2"
 
# install & check on OPAL
python -m pip install opal-optimizer
opal doctor
```
 
</details>
<details>
<summary><b>Aurora</b> (TODO)</summary>
TODO
 
</details>


## References

```bash
@article{bai2026reactive,
  title={Reactive Optimizers},
  author={Bai, Xingjian and Liang, Jia-Hao and Teoh, Jayden and Sitzmann, Vincent},
  journal={arXiv preprint arXiv:2610.33826},
  year={2026}
}
```
