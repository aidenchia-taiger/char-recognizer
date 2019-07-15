# char-recognizer
---
**Character-Level Model for Handwriting Recognition**
This is a model that recognizes handwritten characters in a given document. The full pipeline is as follows:
1. Accept document as input and applies some pre-processing & denoising (not done yet) 
2. Detects text in document & crops them out as sub-images
3. Classifies all cropped out sub-images as either handwritten or digital (not done yet)
4. From a handwritten word image, segments into individual characters and crops them out individually as sub-sub-images
5. Each hadnwritten character sub-sub-image is sent to handwritten recognition model for inference
6. Output is concatenated together and post-processed by a language model (not done yet)
7. All predictions are written to a standard HOCR file format

## Getting Started
---


Model is written in Keras. Please refer to the Confluence [page](https://taiger.atlassian.net/wiki/spaces/NLP/pages/693600765/Word+Recognition+with+Explicit+Character+Segmentation?atlOrigin=eyJpIjoiOTA1YWFmOGUxNDQ4NDk5ZThkZTJlMWUzNTdhNjZlYjYiLCJwIjoiYyJ9) for full explanation.

### Prerequisites
- Ubuntu 16.04
- Python 3.5.2
- Keras 2.2.4
- tensorflow-gpu 1.12.0

## Setup
Run in Terminal (setup your own virtualenv):
`pip install requirements.txt` 

## Train
---
Run in Terminal:
`python3 train.py`

## Test
---
Run in Terminal:
`python3 test.py`

## Demo (Flask UI)
---
Run in Terminal:
`python3 app.py`

## Author
---
Aiden Chia

## TODO
---
- Accept document as input and applies some pre-processing & denoising
- Classifies all cropped out sub-images as either handwritten or digital
- Output is concatenated together and post-processed by a language model

