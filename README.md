# Introduction

本算法接口主要目的用于评估wild场景下获得的图像质量

使用了Musiq算法，该算法在速度+表现上满足目前黄牛项目需求

论文：https://arxiv.org/abs/2108.05997

## How to start

环境配置: 见requirements.txt

## How to use

**you can use it as submodule**

在自己的项目目录下，git submodule add  https://github.com/ahaqu01/image_definition_det.git

便会在项目目录下下载到image_definition_det相关代码

下载完成后，便可在自己项目中使用image_definition_det API，**使用样例和输入输出说明**如下：

```python
from image_definition_det.src.musiq import Musiq
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = Musiq(musiq_config="/workspace/huangniu_demo/image_definition_det/src/configs/musiq.yaml",
          musiq_weights="/workspace/huangniu_demo/image_definition_det/src/weights/musiq_paq2piq_ckpt-364c0c84_2.pth",
          device=device)
res = t.batch_inference(imgs)
# API inputs    								  
    # imgs: list, each item is ndarray,  channel is RGB, shape is (h, w, 3)
    # 设len(imgs)=N

# API outputs
	# res: a tensor, torch.float32, shape is (N,),
    # value range: 0~100, 越大质量越好
    # 可通过卡阈值来筛选删除低质量图片

```

