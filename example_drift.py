from PIL import Image
import glob
import torchvision
from torchvision import transforms, datasets
import json
from tqdm import tqdm
import torch
import pickle
from TextToConcept import TextToConcept
from my_utils import imagenet_classes

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

class ObjectNet(torch.utils.data.Dataset):
    def __init__(self, root='/cmlscratch/mmoayeri/data/objectnet/objectnet-1.0/', 
                 transform=transform,
                 img_format='png'):
        
        self.root = root
        self.transform = transform
        
        files = glob.glob(root+"/**/*."+img_format, recursive=True)
        self.pathDict = {}
        for f in files:
            self.pathDict[f.split("/")[-1]] = f
        self.imgs = list(self.pathDict.keys())
        self.loader = self.pil_loader
        with open(self.root+'mappings/folder_to_onet_id.json', 'r') as f:
            self.folder_to_onet_id = json.load(f)

    def __getitem__(self, index):
        """
        Get an image and its label.
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, onet_id). onet_id is the ID of the objectnet class (0 to 112)
        """
        img, onet_id = self.getImage(index)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, onet_id

    def getImage(self, index):
        """
        Load the image and its label.
        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
        filepath = self.pathDict[self.imgs[index]]
        img = self.loader(filepath)

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)

        # map folder name to objectnet id
        folder = filepath.split('/')[-2]
        onet_id = self.folder_to_onet_id[folder]
        return (img, onet_id)

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
        
if __name__ == '__main__':
    
    model = torchvision.models.resnet50(pretrained=True)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    model.forward_features = lambda x : encoder(x)
    model.get_transform = transform
    
    # loading imagenet dataset to train aligner.
    dset_train = torchvision.datasets.ImageNet(root='/fs/cml-datasets/ImageNet/ILSVRC2012', split='train',)
    
    
    # initiating text-to-concept object.
    text_to_concept = TextToConcept(model)
    text_to_concept.train_linear_aligner(dset_train)
    text_to_concept.save_linear_aligner('imagenet_resnet50_aligner.pth')
    
    
    dset1 = ObjectNet()
    
    with open('/cmlscratch/mmoayeri/data/objectnet/objectnet-1.0/mappings/inet_id_to_onet_id.json', 'r') as f:
        inet_id_to_onet_id = json.load(f)
    sub_idx = []
    for i in inet_id_to_onet_id.keys():
        sub_idx += [500*int(i)+j for j in range(500)]
    
    dset2 = torchvision.datasets.ImageNet(root='/fs/cml-datasets/ImageNet/ILSVRC2012', split='val',)
    dset2 = torch.utils.data.Subset(dset2, sub_idx)
    
    template = 'a photo of a {} indoors'
    
    
    t_tests, _1, _2 = text_to_concept.detect_drift(dset1, dset2,
                                                   list_of_prompts=[template.format(c) for c in imagenet_classes])    

    
                                 