# -*- coding: utf-8 -*-
#dataset ImageNet
#Image size 224x224x3

from modelNet import Net
def test_inference():
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import transforms
    import time

    modelPath = 'modelPath'
    JPGPath = 'C:/Users/user/source/repos/code/gpt/img.jpeg'
    Image = Image.open(JPGPath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = torch.load(modelPath, map_location='cpu')
    model = Net()
    model.eval()
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    start_time = time.time()

    image = test_transforms(Image).float()
    input = image.unsqueeze(0)
    output = model(input)

    finish_time = time.time()

    print((finish_time - start_time))
    print(output)

test_inference()
