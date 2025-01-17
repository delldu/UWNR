# Underwater Light Field Retention : Neural Rendering for Underwater Imaging (UWNR) (Accepted by CVPR Workshop2022 NTIRE)
**<font size=5>Authors:</font>** **Tian Ye<span>&#8224;</span>, Sixiang Chen<span>&#8224;</span>, Yun Liu, Erkang Chen\*, Yi Ye, Yuche Li**

+ **<span>&#8224;</span>**  &ensp;represents equal contributions.
+ **\***  &ensp;represents corresponding author.
<br>

[<font size=4>Paper Download</font>](https://arxiv.org/abs/2203.11006v2)   &emsp; [<font size=4>Code Download</font>](https://github.com/Ephemeral182/UWNR)

**Abstract:** *<font size=2>Underwater Image Rendering aims to generate a true-tolife underwater image from a given clean one, which could be applied to various practical applications such as underwater image enhancement, camera filter, and virtual gaming. We explore two less-touched but challenging problems in underwater image rendering, namely, i) how to render diverse underwater scenes by a single neural network? ii) how to adaptively learn the underwater light fields from natural exemplars, i,e., realistic underwater images? To this end, we propose a neural rendering method for underwater imaging, dubbed UWNR (Underwater Neural Rendering). Specifically, UWNR is a data-driven neural network that implicitly learns the natural degenerated model from authentic underwater images, avoiding introducing erroneous biases by hand-craft imaging models.&nbsp;  
&emsp;&emsp; Compared with existing underwater image generation methods, UWNR utilizes the natural light field to simulate the main characteristics ofthe underwater scene. Thus, it is able to synthesize a wide variety ofunderwater images from one clean image with various realistic underwater images.
&nbsp;  
&emsp;&emsp;  Extensive experiments demonstrate that our approach achieves better visual effects and quantitative metrics over previous methods. Moreover, we adopt UWNR to build an open Large Neural Rendering Underwater Dataset containing various types ofwater quality, dubbed LNRUD.</font>*
##
<p align='center'>
<img src="https://github.com/Ephemeral182/UWNR/blob/master/figure/framework4.png#pic_center" width="80%" ></img>


## Experiment Environment

+ python3 
+ Pytorch 1.9.0
+ Numpy 1.19.5
+ Opencv 4.5.5.62
+ NVDIA 2080TI GPU + CUDA 11.4
+ NVIDIA Apex 0.1
+ tensorboardX(optional)

## Large Neural Rendering Underwater Dataset (LNRUD)

The LNRUD generated by our Neural Rendering architecture can be downloaded from [LNRUD](https://pan.baidu.com/s/1zq8tRxbJb5rUZJbIPtBi9Q) &nbsp; **Password:djhh** , which contains 50000 clean images and 50000 underwater images synthesized from 5000 real underwater scene images.

##
<p align='center'>
<img src="https://github.com/Ephemeral182/UWNR/blob/master/figure/LNRUD.png#pic_center" width="80%" ></img>
                                                                                                  
## Training Stage
All datasets can be downloaded, including [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/) and [SUID](https://ieee-dataport.org/open-access/suid-synthetic-underwater-image-dataset)

Train with the DDP mode under Apex 0.1 and Pytorch1.9.0 

Put clean images in clean_img_path.

Put depth images in depth_img_path.

Put real underwater images as training ground-truth in underwater_path.

Put real underwater images as FID_gt in fid_gt_path.

Run the following commands:
```python
python3  -m torch.distributed.launch --master_port 42563 --nproc_per_node 2 train_ddp.py --resume=True --clean_img_path clean_img_path --depth_img_path depth_img_path --underwater_path underwater_path --fid_gt_path fid_gt_path --model_name UWNR
```
## Generating Stage 

You can download pre-trained model from [Pre-trained model](https://pan.baidu.com/s/1PA6sMo8MIr4HcaPOm6A15g) &nbsp; **Password:42w9** and save it in model_path. The Depth Net refers to [MegaDepth](https://github.com/zhengqili/MegaDepth) and we use the [depth pre-trained model](https://pan.baidu.com/s/1h1qx0Ju7UbyXpjxi2zKeag) &nbsp; **Password:mzqa** from them. 

Run the following commands:
```python
python3  test.py --clean_img_path clean_img_path --depth_img_path depth_img_path --underwater_path underwater_path --fid_gt_path fid_gt_path --model_path model_path 
```
The rusults are saved in ./out/
## Correction
The computation and inferencing runtime of rendering is 138.13GMac/0.026s when the image size is 1024×1024.
## Citation 
```Bibtex
@article{ye2022underwater,
  title={Underwater Light Field Retention: Neural Rendering for Underwater Imaging},
  author={Ye, Tian and Chen, Sixiang and Liu, Yun and Chen, Erkang and Ye, Yi and Li, Yuche},
  journal={arXiv preprint arXiv:2203.11006},
  year={2022}
}
```
If you have any questions, please contact the email 282542428@qq.com or 201921114013@jmu.edu.cn
