# IGNet
[![ACM](https://img.shields.io/badge/ACM-MM2023-purple)](https://www.acmmm2023.org/)
[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/lok-18/IGNet/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

### *Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion (Accept)*
in Proceedings of the 31st ACM International Conference on Multimedia (**ACM MM 2023**)  
by Jiawei Li, Jiansheng Chen, JinyuanLiu and Huimin Ma  
<div align=center>
<img src="https://github.com/lok-18/IGNet/blob/master/fig/1.png" width="100%">
</div>  
Fortunately, our article is in the acceptance list. We will give the article link after the article is published!

### *Requirements* 
> - python 3.7  
> - torch 1.7.0
> - torchvision 0.8.0
> - opencv 4.5
> - numpy 1.21.6
> - pillow 9.4.0

### *Dataset setting*
> We give 5 test image pairs as examples in TNO, MFNet and M3FD datasets, respectively.
>
> Moreover, you can set your own test datasets of different modalities under ```./test_images/...```, like:   
> ```
> test_images
> ├── ir
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ├── vis
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ```
> Dataset download: [[*TNO*]](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) & [[*MFNet*]](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) & [[*M3FD*]](https://github.com/JinyuanLiu-CV/TarDAL)
> 
> Note that if ```./test_images/vis/xxx.png``` is in single-channel L format, you should use ```LtoRGB.py``` to convert it to three-channel RGB format.
