# ImageDetectors

### Target Completion Date: Saturday or Sunday

### TODO list: 
1. [x] Face Detection + Registration
2. [x] Model Pretraining*
3. [x] Build training sets and test sets for CK/CK+.
4. [ ] Fine-tune base model on CK/CK+
5. [x] Map FACS to emojis
6. [x] Implement Emoji Matching
7. [ ] Make a live-camera interface to display both face and predicted emoji**

\*  Use CelebA database (~100,000 images) and run each face through OpenFace AU predictor to obtain noisy labels for faces, then train on pretrained VGG-13 model to obtain base model. Training/Test set should be divided based on gender to prevent overfitting.
** If we don't get this far, we could just stick with static images.

## Tentative Model Pipeline

##### Preprocessing: 

- *Face Detection*: Either use HOG+SVM or Viola-Jones to get face bounding boxes. Dlib has HOG+SVM builtin while OpenCV has Viola-Jones built in.
- *Face Registration*: Use ensemble of regression trees to obtain face landmarks (built-in in Dlib). Get eye landmarks and rotate face such that line connecting eye landmarks are horizontal so face orientation is normalized. Crop face into 224x224 image. See https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ for implementation details.
- Optionally, could use MTCNN to get both face landmarks and face bounding boxes together.
- *Final Preprocessing*: Convert image to grayscale and mean normalize so that images have mean 0 and standard deviation 1.

##### Feature Extraction + Classification:

- Do feature extraction and classification together through a Convolutional Neural Network. Use VGG-13 model that we fine-tune with CK and CK+ but replace output layer with an output layer of 25 units (1 for each action unit). Each unit should have an output between 0 and 1 representing the AU intensity.

##### Emoji Matching:
- Output emoji such that labels for emoji and classifier output maximize Kappa coefficient (measure of agreement between emoji labels and classifier output) over all emojis. 

### Interesting Materials
- https://paperswithcode.com/paper/au-r-cnn-encoding-expert-prior-knowledge-into
- https://arxiv.org/pdf/1904.01509.pdf - Though we can't use this dataset, it provides a good baseline model we can use
- https://arxiv.org/pdf/1911.05946.pdf - We can use the pretraining part to make up for the fact that we don't have much accurately labeled data to work with
