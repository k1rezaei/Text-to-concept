# Text-To-Concept
We implemented class `TextToConcept` which includes implementation of following functions:
+ `TextToConcept(model)`: <b>initiating TextToConcept object.</b>
  Inputs:
  + A torch object `model`  which implements following functions and attributes:
    + ``forward_features(x)`` that takes a tensor as the input and outputs the representation (features) of input $x$ when it is passed through the model.
    + ``get_normalizer`` should be the normalizer that the models uses to preprocess the input. e.g., Resnet18, uses standard ImageNet normalizer.
    + Attribute ``has_normalizer`` should be `True` when normalizer is need for the model.


  Output: Text-To-Concept object.

+ `train_linear_aligner(D, save_reps, load_reps, path_to_model, path_to_clip_model, epochs)`: <b>training linear aligner.</b>
  Inputs:
    + `D`: the dataset that linear aligner should be trained on (e.g., ImageNet). Dataset is supposed to have tensors with standard ImageNet size. Normalization must not be done on `D`.
    + `save_reps`: boolean that determines whether representation spaces of model and clip model should be stored or not (`False` by default).
    + `load_reps`: boolean that determines whether representation spaces of model and clip model can be loaded or not (`False` by default).
      + if this boolean is `False`, we obtain representations which takes additional time.
    + `path_to_model` if at least one `save_reps` or `load_reps` is `True`, this argument shows the path to save/load model representation space.
    + `path_to_clip_model` if at least one `save_reps` or `load_reps` is `True`, this argument shows the path to save/load clip model representation space.


  Outputs: this function doesn't return anything but the linear aligner from `model` space to vision-language space is trained when execution of this function is done.

+ `save_linear_aligner(path_to_save)`: <b>saving linear aligner.</b>
  Inputs:
    + `path_to_save`: this argument shows the path to save linear aligner.


  Outputs: None

+ `load_linear_aligner(path_to_load)`: <b>saving linear aligner.</b>
  Inputs:
    + `path_to_load`: this argument shows the path to load linear aligner.


  Outputs: this function doesn't return anything but the linear aligner is loaded when execution of this function is done.

+ `get_zero_shot_classifier(classes, prompts)`: <b>getting a zero-shot classifier for arbitrary classification problem.</b>
  Inputs:
  + `classes`: a list contaning class names. e.g., ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck').
  + `prompts`: a list of prompts such that each prompt takes class names as input. e.g., ('a photo of a {}', 'a pixelated photo of a {}', 'a photo of the large {}').
    + `prompt.format(c)` for `prompt` in `prompts` and `c` in `classes` should generate a valid explanation of the class.


  Outputs:
  + This function returns an object $f$ of the class `ZeroShotClassifier` such that $f(x)$ returns the logits of classifier when input $x$ is given.
    + Note that $x$ should be in a way that $\textnormal{model}(x)$ generates the logits of the original model when input $x$ is passed through the model.


+ `search(dset, prompts)`: <b>searching for images in `dset` which match prompts.</b>
  Inputs:
  + `dset`: the dataset on which we do retrieval. it should contain tensors with standard ImageNet size and not normalized.
  + `prompts`: a list of prompts. these prompts should generally describe the same object.


  Outputs: this function returns a pair `(indices, sims)`.
  + `indices`: a list including all image indices sorted based on the similiarity to the text prompts.
  + `sims`: a list including similarity of images and text prompts. More precisely, `sims[i` is the similarity of image `i` to prompts.


+ `concept_logic(dset, list_of_prompts, signs, scales)`: <b>searching for particular images in `dset` with logical expressions.</b>
  Inputs:
  + `dset`: the dataset on which we do retrieval. it should contain tensors with standard ImageNet size and not normalized.
  + `list_of_prompts`: a list of prompts. Each element in the list corresponds to a concept and includes prompts explaining that.
  + `signs`: list of signs. Each sign (+1/-1) correponds to a concept and shows whether we look for concept's presence or absence.
  + `scales`: list of scales. Each scale correponds to a concept and shows the degree of intense for that concept.


  Outputs: this function returns a pair `(indices, sims)`.
  + `indices`: a list including indices of images which satisfy concept logic requirements.
  + `sims`: a list including similarity of images and different concepts. More precisely, `sims[i, j]` is the similarity of image `i` to concept `j`.

+ `detect_drift(dset1, dset2, list_of_prompts)`: <b>t-test for distribution shift w.r.t to different concepts between two datasets.</b>
  Inputs:
  + `dset1`: first dataset. it should contain tensors with standard ImageNet size and not normalized.
  + `dset2`: second dataset. it should contain tensors with standard ImageNet size and not normalized.
  + `list_of_prompts`: a list of prompts. Each element in the list corresponds to a concept and includes prompts explaining that.

  Outputs: this function returns a pair `(t-tests, sims1, sims2)`.
  + `t-tests`: a list including t-test results `(t-stats, p-value)` for each of the concepts.
  + `sims1`: a list including similarity of images in `dset1` and different concepts. More precisely, `sims1[i, j]` is the similarity of image `i` to concept `j`.
  + `sims2`: a list including similarity of images in `dset2` and different concepts. More precisely, `sims2[i, j]` is the similarity of image `i` to concept `j`.


We refer to [notebook1](example_search_concept_logic.ipynb) and [notebook2](example_zeroshot.ipynb) for more details on how to use Text-To-Concept framework.




