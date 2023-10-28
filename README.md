# IGNet
[![ACM](https://img.shields.io/badge/ACM-MM2023-purple)](https://www.acmmm2023.org/)
[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/lok-18/IGNet/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

### *Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion (Accept)*
in Proceedings of the 31st ACM International Conference on Multimedia (**ACM MM 2023**)  
by Jiawei Li, Jiansheng Chen, JinyuanLiu and Huimin Ma 
- [[*ACM DL*]](https://dl.acm.org/doi/10.1145/3581783.3612135)
- [[*arXiv*]](https://arxiv.org/abs/2308.03256)
- [[*Google Scholar*]](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Learning+a+Graph+Neural+Network+with+Cross+Modality+Interaction+for+Image+Fusion&btnG=)

**Overall performance comparison:**
<div align=center>
<img src="https://github.com/lok-18/IGNet/blob/master/fig/compare.png" width="100%">
</div>  


**Framework of our proposed IGNet:**
<div align=center>
<img src="https://github.com/lok-18/IGNet/blob/master/fig/network.PNG" width="100%">
</div> 


<br>**GIF Demo:**</br>
<div align=center>
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_ir1.gif" width="15.5%" > 
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_vis1.gif" width="15.5%">
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_f1.gif" width="15.5%">
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_ir2.gif" width="15.5%">
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_vis2.gif" width="15.5%">
<img src="https://github.com/lok-18/IGNet/blob/master/fig/demo_f2.gif" width="15.5%">
</div> 

### *Requirements* 
> - python 3.7  
> - torch 1.7.0
> - torchvision 0.8.0
> - opencv 4.5
> - numpy 1.21.6
> - pillow 9.4.0

### *Dataset setting*
> We give 5 test image pairs as examples in [[*TNO*]](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029), [[*MFNet*]](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) and [[*M3FD*]](https://github.com/JinyuanLiu-CV/TarDAL) datasets, respectively.
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
> 
> Note that if ```./test_images/vis/xxx.png``` is in single-channel L format, you should use ```LtoRGB.py``` to convert it to three-channel RGB format.

### *Test*
> The pre-trained model has given in ```./model/IGNet.pth```.
> Please run ```test.py``` to get fused results, and you can check them in:
> ```
> results
> ├── 1.png
> ├── 2.png
> └── ...

### *Experimental results*
> We compared our proposed IGNet with [[*DIDFuse*]](https://github.com/Zhaozixiang1228/IVIF-DIDFuse), [[*U2Fusion*]](https://github.com/hanna-xu/U2Fusion), [[*SDNet*]](https://github.com/HaoZhang1018/SDNet), [[*TarDAL*]](https://github.com/JinyuanLiu-CV/TarDAL), [[*UMFusion*]](https://github.com/wdhudiekou/UMF-CMGR), [[*DeFusion*]](https://github.com/erfect2020/DecompositionForFusion) and [[*ReCoNet*]](https://github.com/JinyuanLiu-CV/ReCoNet).
> 
> **Fusion results:**
> <div align=center>
> <img src="https://github.com/lok-18/IGNet/blob/master/fig/fusion.png" width="100%">
> </div>
>
> <br>After retaining the fusion results of all methods on [[*YOLOv5*]](https://github.com/ultralytics/yolov5) and [[*DeepLabV3+*]](https://github.com/VainF/DeepLabV3Plus-Pytorch), we compare the corresponding detection and segmentation results with IGNet.</br>
> 
> **Detection & Segmentation results:**
> <div align=center>
> <img src="https://github.com/lok-18/IGNet/blob/master/fig/detection.png" width="100%">
> </div>
> <div align=center>
> <img src="https://github.com/lok-18/IGNet/blob/master/fig/segmentation.png" width="100%">
> </div>
> Please refer to the paper for more experimental results and details.
>
### *Citation*
> ```
> @inproceedings{li2023learning,
>    title = {Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion},
>    author = {Li, Jiawei and Chen, Jiansheng and Liu, Jinyuan and Ma, Huimin},
>    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
>    pages = {4471–4479},
>    year = {2023},
>    doi = {10.1145/3581783.3612135}
> }
> ```
>
### *Realted works*
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***GeSeNet: A General Semantic-guided Network with Couple Mask Ensemble for Medical Image Fusion***. IEEE Transactions on Neural Networks and Learning Systems (**IEEE TNNLS**), 2023. [[*Paper*]](https://ieeexplore.ieee.org/document/10190200) [[*Code*]](https://github.com/lok-18/GeSeNet)
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***Learning a Coordinated Network for Detail-refinement Multi-exposure Image Fusion***. IEEE Transactions on Circuits and Systems for Video Technology (**IEEE TCSVT**), 2022, 33(2): 713-727. [[*Paper*]](https://ieeexplore.ieee.org/abstract/document/9869621)
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***Infrared and visible image fusion based on residual dense network and gradient loss***. Infrared Physics & Technology, 2023, 128: 104486. [[*Paper*]](https://www.sciencedirect.com/science/article/pii/S1350449522004674)
> - Jia Lei, Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***GALFusion: Multi-exposure Image Fusion via a Global-local Aggregation Learning Network***. IEEE Transactions on Instrumentation and Measurement (**IEEE TIM**), 2023, 72: 1-15. [[*Paper*]](https://ieeexplore.ieee.org/abstract/document/10106641) [[*Code*]](https://github.com/lok-18/GALFusion)
>
### *Acknowledgement*
>Our Code is partially adapted from [[*Cas-Gnn*]](https://github.com/LA30/Cas-Gnn). Please refer to their excellent work for more details.
> 
### *Contact*
> If you have any questions, please create an issue or email to me ([Jiawei Li](mailto:ljw19970218@163.com)).
