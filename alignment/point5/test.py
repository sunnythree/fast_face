
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as tfs
from point5.model import CnnAlign
from point5.dataset import MTFLDataset, draw_ann, de_normal_anns
import PIL.ImageFont as ImageFont
import numpy as np
import time


MODEL_FACE_ALIGN = "./output/alignment520.pt"

font_size = 4
font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)


def test():
    data_loader = DataLoader(dataset=MTFLDataset(False, 64), batch_size=1, shuffle=True, num_workers=1)
    device = torch.device("cpu")
    model = CnnAlign().to(device)
    state = torch.load(MODEL_FACE_ALIGN, map_location='cpu')
    model.load_state_dict(state['net'])
    to_pil_img = tfs.ToPILImage()
    model.eval()
    for img, label in data_loader:
        start = time.time()
        output = model(img)
        end = time.time()
        cost = (end - start)
        print("cost : " + str(cost))
        pil_img = to_pil_img(img[0].cpu())
        anns = output[0].cpu().detach().numpy()
        anns = np.resize(anns, (10, 2))
        anns = de_normal_anns(anns, 64, 64)
        draw_ann(pil_img, anns.tolist(), font1, font_size)
        plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        plt.imshow(pil_img)
        plt.show()
        plt.close()

if __name__ == '__main__':
    test()

