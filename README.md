# Character-Level Model for Handwriting Recognition

## Background
![Text Detected](../figures/textDetected.png)
This is a model that recognizes handwritten characters in a given document. The full pipeline is as follows:
1. Accept document as input and applies some pre-processing & denoising (not done yet) 
2. Detects text in document & crops them out as sub-images
3. Classifies all cropped out sub-images as either handwritten text, digital text, or just noise (not done yet)
4. From a handwritten word image, segments into individual characters and crops them out individually as sub-sub-images
5. Each handwritten character image is sent to handwritten recognition model for inference
6. Output is concatenated together and post-processed by a language model (not done yet)
7. All predictions are written to a standard HOCR file format

Handwriting Recognition Model is a deep Convolutional Neural Network written in Keras. Please refer to the Confluence [page](https://taiger.atlassian.net/wiki/spaces/NLP/pages/693600765/Word+Recognition+with+Explicit+Character+Segmentation?atlOrigin=eyJpIjoiOTA1YWFmOGUxNDQ4NDk5ZThkZTJlMWUzNTdhNjZlYjYiLCJwIjoiYyJ9) for a full explanation.



## Getting Started

### Environment
- Ubuntu 16.04
- Python 3.5.2
- Keras 2.2.4
- tensorflow-gpu 1.12.0

### Prerequisites
Run in Terminal (setup your own virtualenv):
`pip install -r requirements.txt` 

### Train model
Under `src` directory:   
`python3 train.py --model MODEL_NAME`   

The images used for training should be placed under `../imgs/train`, and those used for validation should be placed under `../imgs/validation`. While training, you can view progress on Tensorboard. Saved model will be automatically saved as ../models/MODEL_NAME.h5.

### Test on a directory of character images
Under `src` directory:   
`python3 test.py --model MODEL_NAME --test PATH_TO_DIR`   

`PATH_TO_DIR` should be a valid path to a directory of character-level images and `MODEL_NAME` should be the name of a trained model under the `../models` directory.

### Run inference on any given document, word, or character image
Under `src` directory:      
`python3 main.py --model MODEL_NAME --type TYPE -- infer PATH_TO_IMAGE`

`TYPE` can be either `doc`, `word`, or `char`, depending on the type of image you're inferring on. HOCR output will be written to working directory as `output.hocr` by default.

### Demo (Flask UI)
Under `src` directory:    
`python3 app.py --model MODEL_NAME`    

`MODEL_NAME` should be the name of a trained model under the `../models` directory.

### Things to Note
1. The `--model` command line argument should be the name of a model, not a path. Suppose `alpha.h5` is a model under the `models` directory, then the correct command line argument should be `--model alpha`, NOT `--model ../alpha.h5` or `--model alpha.h5`
2. You can check all command line arguments for each file by specifying `-h`
3. The `test.py` file is used to test the performance of the CNN itself, and hence it will not work on a directory of word or document images

## Author
Aiden Chia
You may reach me via Slack.

## Todo
- Accept document as input and applies some pre-processing & denoising
- Classifies all cropped out sub-images as either handwritten or digital
- Output is concatenated together and post-processed by a language model
- Implement evaluation for `test.py` file