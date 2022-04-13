# UWNR

# Underwater Light Field Retention : Neural Rendering for Underwater Imaging (UWNR) (Accepted by CVPR2022 Workshop)
**<font size=5>Authors:</font>** **Tian Ye\*, Sixiang Chen\*, Yun Liu, Erkang Chen\**, Yi Ye, Yuche Li**

+ **\***  &nbsp;&ensp; represents equal contributions.
+ **\****  &ensp;represents corresponding author.
<br>

[<font size=4>Paper Download</font>](https://arxiv.org/pdf/2203.11006.pdf)   &emsp; [<font size=4>Code Download</font>](https://github.com/Ephemeral182/UWNR)

**Abstract:** *<font size=2>Underwater Image Rendering aims to generate a true-tolife underwater image from a given clean one, which could be applied to various practical applications such as underwater image enhancement, camera filter, and virtual gaming. We explore two less-touched but challenging problems in underwater image rendering, namely, i) how to render diverse underwater scenes by a single neural network? ii) how to adaptively learn the underwater light fields from natural exemplars, i,e., realistic underwater images? To this end, we propose a neural rendering method for underwater imaging, dubbed UWNR (Underwater Neural Rendering). Specifically, UWNR is a data-driven neural network that implicitly learns the natural degenerated model from authentic underwater images, avoiding introducing erroneous biases by hand-craft imaging models.&nbsp;  
&emsp;&emsp; Compared with existing underwater image generation methods, UWNR utilizes the natural light field to simulate the main characteristics ofthe underwater scene. Thus, it is able to synthesize a wide variety ofunderwater images from one clean image with various realistic underwater images.
&nbsp;  
&emsp;&emsp;  Extensive experiments demonstrate that our approach achieves better visual effects and quantitative metrics over previous methods. Moreover, we adopt UWNR to build an open Large Neural Rendering Underwater Dataset containing various types ofwater quality, dubbed LNRUD.</font>*

![1649767279060.png](./img/1649767279060.png)
## Installation 
-----
+ python3 
+ Pytorch 1.9.0
+ NVDIA 2080TI GPU + CUDA 11.4
+ Apex 0.1
+ tensorboardX(optional)

## Large Neural Rendering Underwater Dataset (LNRUD)



## Code 

Train with the DDP mode under Apex 0.1 and Pytorch1.9.0

Run the following commands:
```python
python3  -m torch.distributed.launch --master_port 42563 --nproc_per_node 2 train.py --resume=True
```
## Citation 
```Bibtex
@article{ye2022underwater,
  title={Underwater Light Field Retention: Neural Rendering for Underwater Imaging},
  author={Ye, Tian and Chen, Sixiang and Liu, Yun and Chen, Erkang and Ye, Yi and Li, Yuche},
  journal={arXiv preprint arXiv:2203.11006},
  year={2022}
}
```
