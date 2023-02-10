

class C2T:
    
    # initialization the object
    # (f, D, clip-model, ...)
    def __init__(self, f, D, clip_model) -> None:
        # get ftr representation of images in D
        #   in both model f's space and clip_model space.
        # instantiate the Linear Aliger object and learn W, b.
        pass
    
    
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

