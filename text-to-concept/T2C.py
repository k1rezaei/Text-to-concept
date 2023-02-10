import torch
from torchvision import datasets, transforms, models
import torchvision
import open_clip
from tqdm import tqdm
import numpy as np
from LinearAligner import LinearAligner


class T2C:
    
    # initialization the object
    # (f, D, clip-model, ...)
    def __init__(self, f, D, clip_model) -> None:
        reps_f, logits_f = self.obtain_ftrs(f, D) # TODO: we may discard logits
        reps_clip, logits_clips = self.obtain_ftrs(clip_model, D, apply_norm=False) # TODO: we may discard logits
    
        self.linear_aligner = LinearAligner()
        self.linear_aligner.train(reps_f, reps_clip, epochs=5, target_variance=4.5, verbose=1)
        # this linear aligner has get_aligned_representation function which mught be useful.
    
    # get zero-shot classifier based on text prompts and templates.
    def zero_shot_classifier(list_of_text_prompts, list_of_templates):
        # we should return a function which implements the process.
        pass
    
    
    # search for top k images in dataset D, based on text prompt
    def search(D, text_propmt, template):    
        pass
    
    
    # detect if there is a shift in distribution for some concepts.
    def detect_drift(D1, D2, list_of_text_prompts, list_of_templates):
        # we should compare distributions D1 and D2 based on list of concepts we have.
        # result of this function includes whether significant drift is recognized or not (and corresponding z-score for example).
        # plots for each concept might be helpful as well...
        pass
    
    # search for top k images with a logical expression.
    # Thresholds can be found using the initial distribution, or can be provided by the user.
    def concept_logic(list_of_text_prompts, list_of_templates, list_of_thresholds, list_of_negations):
        pass
    
    # we may also have several utility functions private for this class as well.
    
    
    def obtain_ftrs(self, model, D, apply_norm=True):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
        
        # TODO? what are our assumptions on D?
        dset = datasets.ImageNet(root='/fs/cml-datasets/ImageNet/ILSVRC2012', split='train', transform=self.transform)
        loader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True) 
                                
        return self.obtain_reps_given_loader(model, loader, apply_norm=True, samples_per_class=250)
    
    
    def obtain_reps_from_batch(self, model, imgs, labels, apply_norm):
        subset = imgs.cuda()
        
        if apply_norm:
            subset = self.normalize(subset) # TODO do we need this?
            
        reps = model.get_reps(subset).flatten(1)
        
        logits = model.head(reps) # TODO do we need logits?
        reps, logits = [x.detach().cpu().numpy() for x in [reps, logits]]
        return reps, logits


    def obtain_reps_given_loader(self, model, loader, apply_norm=True, samples_per_class=250):
        all_reps, all_logits = [], []

        for imgs, labels in tqdm(loader):
            if sum(labels == curr_class) == 0:
                continue
            reps, logits = self.obtain_reps_from_batch(model, imgs, labels, apply_norm)
            all_reps.extend(reps)
            all_logits.extend(logits)
            cnt_in_curr_class += reps.shape[0]

            if cnt_in_curr_class == samples_per_class or cnt_in_curr_class == 0:
                cnt_in_curr_class = 0
                curr_class += 1
                if sum(labels == curr_class) == 0:
                    continue
                reps, logits = self.obtain_reps_from_batch(model, imgs, labels, apply_norm)
                all_reps.extend(reps)
                all_logits.extend(logits)
                cnt_in_curr_class += reps.shape[0]

        all_reps, all_logits = [np.stack(x) for x in [all_reps, all_logits]]
        return all_reps, all_logits
