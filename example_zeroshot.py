import torch
import torchvision
import numpy as np
from tqdm import tqdm

from TextToConcept import TextToConcept

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # resnet18 model.
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    
    model = torchvision.models.resnet18(pretrained=True)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    model.forward_features = lambda x : encoder(x)
    model.get_transform = transform
    
    # loading imagenet dataset to train aligner.
    dset = torchvision.datasets.ImageNet(root='/fs/cml-datasets/ImageNet/ILSVRC2012', split='train',)
    
    # initiating text-to-concept object.
    text_to_concept = TextToConcept(model)
    text_to_concept.train_linear_aligner(dset)
    text_to_concept.save_linear_aligner('imagenet_resnet18_aligner.pth')
    
    # loading SVHN.
    svhn = torchvision.datasets.SVHN(root='data/', split='test', download=True, transform=transform)
    svhn.classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # getting svhn zero-shot classifier.
    svhn_zeroshot_classifier = text_to_concept.get_zero_shot_classifier(svhn.classes, prompts=['a photo of a {}'])
    
    # accuracy on testdata.
    
    loader = torch.utils.data.DataLoader(svhn, batch_size=4, shuffle=True, num_workers=8)
    correct, total = 0, 0
    for data in tqdm(loader):
        x, y = data[:2]
        x = x.to(device)
        y = y.to(device)
        
        _, predicted = svhn_zeroshot_classifier(x).max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
    print(f'ResNet18 Zeroshot Accuracy on SVHN {100.*correct/total:.2f}')
        


    