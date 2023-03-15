from PIL import Image
import os
import yaml
import torch
import torchvision
from .models.musiq_arch import MUSIQ


class Musiq(object):
    def __init__(self,
                 musiq_config="",
                 musiq_weights="",
                 device=None,
                 trans=None):

        with open(musiq_config, 'r', encoding='utf-8') as mu_f:
            cont = mu_f.read()
            mu_cfg = yaml.load(cont)
            self.mu_cfg = mu_cfg
        self.mu_weights = musiq_weights
        self.device = device

        # define transforms
        if trans is None:
            self.img_range = self.mu_cfg["musiq"]["img_range"]
            self.input_size = self.mu_cfg["musiq"]["input_size"]
            tf_list = []
            tf_list.append(torchvision.transforms.Resize(self.input_size))
            tf_list.append(torchvision.transforms.ToTensor())
            tf_list.append(torchvision.transforms.Lambda(lambda x: x * self.img_range))
            self.trans = torchvision.transforms.Compose(tf_list)
        else:
            self.trans = trans

        # create model
        longer_side_lengths = self.mu_cfg["musiq"]["longer_side_lengths"]
        self.model = MUSIQ(longer_side_lengths=longer_side_lengths)
        if os.path.exists(self.mu_weights):
            self.load_weights()
        else:
            raise IOError("checkpoint_dir not exists!")
        if self.device is not None:
            self.model.to(self.device)
        self.model.eval()

    def load_weights(self):
        could_load = self._load_checkpoint(self.mu_weights)
        if could_load:
            print('Checkpoint load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.model.load_state_dict(checkpoint)
            return True
        else:
            return False

    def pre_process(self, image_batch_list):
        batch_tensor = []
        for img in image_batch_list:
            img_tensor = self.trans(Image.fromarray(img).convert("RGB"))
            batch_tensor.append(img_tensor)

        batch_tensor = torch.stack(batch_tensor, dim=0).to(self.device)
        return batch_tensor

    @torch.no_grad()
    def batch_inference(self, image_batch_list):
        # image_batch_list: list, item is ndarray. channel:rgb
        batch_tensor = self.pre_process(image_batch_list)
        scores = self.model(batch_tensor)
        return scores


if __name__ == '__main__':
    import os
    import time
    import numpy as np
    import xlwt

    test_img_root = "/workspace/huangniu_demo/image_definition_det/test_data_blured"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = Musiq(musiq_config="/workspace/huangniu_demo/image_definition_det/src/configs/musiq.yaml",
              musiq_weights="/workspace/huangniu_demo/image_definition_det/src/weights/musiq_paq2piq_ckpt-364c0c84_2.pth",
              device=device)
    image_batch_list = []
    for img_name in os.listdir(test_img_root):
        img_path = os.path.join(test_img_root, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        image_batch_list.append(img)
        print(img_path)
    # res = t.batch_inference(image_batch_list[:10])
    # print(res)
    s_t = time.time()
    for i in range(100):
        res = t.batch_inference(image_batch_list[:50])
        print(i, res)
    e_t = time.time()
    print(res)
    print((e_t - s_t) / 100.)
