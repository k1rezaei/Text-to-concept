import torch
import torchvision
import numpy as np
from tqdm import tqdm

from TextToConcept import TextToConcept

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    model = torchvision.models.resnet18(pretrained=True)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    model.forward_features = lambda x : encoder(x)
    model.get_transform = transform
    
    text_to_concept = TextToConcept(model)
    text_to_concept.load_linear_aligner('imagenet_resnet18_aligner.pth')
    
    # loading imagenet testdata
    dset = torchvision.datasets.ImageNet(root='/fs/cml-datasets/ImageNet/ILSVRC2012', split='train', transform=transform)
    
    sorted_inds, sims = text_to_concept.search(dset=dset,
                                               prompts=['a photo of a bear in a tree', 'a photo of bear in forest'])
    # visualizng two images ...
    
    templates = ['itap of a {}', 'a bad photo of the {}', 'a origami {}', 'a photo of the large {}',
             'a {} in a video game', 'art of the {}', 'a photo of the small {}']
    
    indices, _ = text_to_concept.concept_logic(dset=dset,
                                               list_of_prompts=[[template.format(c) for template in templates] for c in ['a dog', 'the beach', 'the sunset']],
                                               signs=[1, 1, 1],
                                               scales=[2.25, 2, 2])
    
    indices, _ = text_to_concept.concept_logic(dset=dset,
                                               list_of_prompts=[[template.format(c) for template in templates] for c in ['a cat', 'orange', 'indoors']],
                                               signs=[1, 1, -1],
                                               scales=[3, 2, 0])
    

