from typing import Any
import torch
from torchvision import datasets, transforms, models
import torchvision
from tqdm import tqdm
import numpy as np
from LinearAligner import LinearAligner
import clip
import scipy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORMATION = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])
CLIP_IMAGENET_TRANSFORMATION = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

class ClipZeroShot(torch.nn.Module):
    def __init__(self, mtype):
        super(ClipZeroShot, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = transforms.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False
        
    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)
        return image_features
    
    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)


class ZeroShotClassifier:
    def __init__(self, model, aligner: LinearAligner, zeroshot_weights: torch.Tensor):
        self.model = model
        self.aligner = aligner
        self.zeroshot_weights = zeroshot_weights.float()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    # this functions returns logits.
    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            reps = self.model.forward_features(x.to(self.device)).flatten(1)
            aligned_reps = self.aligner.get_aligned_representation(reps)
            aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)
            return aligned_reps @ self.zeroshot_weights.T
        
class TextToConcept:
    # model.forward_features(), model.get_normalizer() should be implemented.
    def __init__(self, model,) -> None:
        
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_model = ClipZeroShot('ViT-B/16')
        
        self.model.eval().to(self.device)
        self.clip_model.eval().to(self.device)
    
    
    def load_linear_aligner(self, path_to_load):
        self.linear_aligner = LinearAligner()
        self.linear_aligner.load_W(path_to_load)
    
    
    def save_reps(self, path_to_model, path_to_clip_model):
        print(f'Saving representations')
        np.save(path_to_model, self.reps_model)
        np.save(path_to_clip_model, self.reps_clip)    
    
    def load_reps(self, path_to_model, path_to_clip_model):
        print(f'Loading representations ...')
        self.reps_model = np.load(path_to_model)
        self.reps_clip = np.load(path_to_clip_model)
    
    def train_linear_aligner(self, D, save_reps=False, load_reps=False, path_to_model=None, path_to_clip_model=None, epochs=5):
        if load_reps:
            self.load_reps(path_to_model, path_to_clip_model)
        else:
            print(f'Obtaining representations ...')
            self.reps_model = self.obtain_ftrs(self.model, D)
            self.reps_clip = self.obtain_ftrs(self.clip_model, D)

        if save_reps:
            self.save_reps(path_to_model, path_to_clip_model)
            
        self.linear_aligner = LinearAligner()
        self.linear_aligner.train(self.reps_model, self.reps_clip, epochs=epochs, target_variance=4.5,)
        
    def save_linear_aligner(self, path_to_save):
        self.linear_aligner.save_W(path_to_save)
        
    def get_zeroshot_weights(self, classes, prompts):
        zeroshot_weights = []
        for c in classes:
            tokens = clip.tokenize([prompt.format(c) for prompt in prompts])
            c_vecs = self.clip_model.encode_text(tokens.to(self.device))
            c_vec = c_vecs.mean(0)
            c_vec /= c_vec.norm()
            zeroshot_weights.append(c_vec)
        
        return torch.stack(zeroshot_weights)
    
    
    def get_zero_shot_classifier(self, classes, prompts=['a photo of {}.']):
        return ZeroShotClassifier(self.model, self.linear_aligner, self.get_zeroshot_weights(classes, prompts))
    
    
    def search(self, dset, prompts=['a photo of a dog']):    
        tokens = clip.tokenize(prompts)
        vecs = self.clip_model.encode_text(tokens.to(self.device))
        vec = vecs.detach().mean(0).float().unsqueeze(0)
        vec /= vec.norm()
        sims = self.get_similarity(dset, self.model.has_normalizer, vec)[:, 0]
        return np.flip(np.argsort(sims,)), sims
        

    def get_similarity(self, dset, do_normalization, vecs: torch.Tensor):
        loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True) 
        all_sims = []
        with torch.no_grad():
            for data in tqdm(loader):
                imgs = data[0]
                if do_normalization:
                    imgs = self.model.get_normalizer(imgs).to(self.device)
                else:
                    imgs = imgs.to(self.device)
                
                reps = self.model.forward_features(imgs).flatten(1)
                aligned_reps = self.linear_aligner.get_aligned_representation(reps)
                aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)
                sims = aligned_reps @ vecs.T
                sims = sims.detach().cpu().numpy()
                all_sims.append(sims)
            
        return np.vstack(all_sims)

    
    def encode_text(self, list_of_prompts):
        all_vecs = []
        for prompts in list_of_prompts:
            tokens = clip.tokenize(prompts)
            vecs = self.clip_model.encode_text(tokens.to(self.device))
            vec = vecs.mean(0).float()
            vec /= vec.norm()
            all_vecs.append(vec)
        
        return torch.stack(all_vecs)
        

    def detect_drift(self, dset1, dset2, list_of_prompts):
        vecs = self.encode_text(list_of_prompts)
        sims1, sims2 = self.get_similarity(dset1, self.model.has_normalizer, vecs), self.get_similarity(dset2, do_transform, vecs)
        
        num_of_concepts = vecs.shape[0]
        
        t_tests = []
        for j in range(num_of_concepts):
            stats, p_value = scipy.stats.ttest_ind(sims1[:, j], sims2[:, j])
            t_tests.append([stats, p_value])
        
        return np.array(t_tests), sims1, sims2
        
        
    def concept_logic(self, dset, list_of_prompts, signs, scales):
        vecs = self.encode_text(list_of_prompts)
        sims = self.get_similarity(dset, self.model.has_normalizer, vecs)
        means = np.mean(sims, axis=0)
        stds = np.std(sims, axis=0)
        
        ths = means + np.array(signs) * np.array(scales) * stds
        retrieved = np.arange(sims.shape[0])
        for j in range(len(signs)):
            if retrieved.shape[0] == 0:
                break
            
            sim_to_concept = sims[retrieved, j]
            if signs[j] == -1:
                retrieved = retrieved[np.where(sim_to_concept < ths[j])[0]]
            else:
                retrieved = retrieved[np.where(sim_to_concept > ths[j])[0]]
        
        return retrieved, sims
        
        
    def obtain_ftrs(self, model, dset):
        loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True) 
        return self.obtain_reps_given_loader(model, loader)
    
    
    def obtain_reps_given_loader(self, model, loader):
        all_reps = []
        for imgs, _ in tqdm(loader):
            if model.has_normalizer:
                imgs = model.get_normalizer(imgs)
            
            imgs = imgs.to(self.device)
                
            reps = model.forward_features(imgs).flatten(1)
            reps = [x.detach().cpu().numpy() for x in reps]
            
            all_reps.extend(reps)
            
        all_reps = np.stack(all_reps)
        return all_reps


