# Image Search with CLIP

## Overview
This project demonstrates **text-to-image** and **image-to-image search** using OpenAI's pretrained CLIP ViT-B/32 model through Hugging Face Transformers. It encodes a small collection of Food-101 images into feature vectors, then compares them with either a text query or a query image. Similarity scores are rescaled using CLIP's logit scale and passed through softmax to rank the collection images.

## Features
- Load the pretrained `openai/clip-vit-base-patch32` model, tokenizer, and image processor.
- Encode local images into normalized feature vectors.
- Search images using a text description.
- Search for visually similar images using a query image.
- Compare query features against the image collection with matrix multiplication.

## Requirements
- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Pillow

Install the dependencies:

```bash
pip install torch transformers pillow
```

## Dataset and file paths
The notebook expects Food-101 images to be available locally. The example paths are:

```text
data/food-101/images/chocolate_cake/13907.jpg
data/food-101/images/chicken_wings/490.jpg
data/food-101/images/chocolate_cake/105067.jpg
```

Update `imagePaths_data` and `imagePaths_query` in the notebook to match the locations of your own images. The notebook does not download or prepare the Food-101 dataset automatically.

## Run
1. Clone or download this repository and open `search_from_images.ipynb` in Jupyter Notebook or VS Code.
2. Install the dependencies listed above.
3. Ensure the image files exist at the paths configured in the notebook.
4. Run the notebook cells in order.
5. Change `query` to search using another text description, or change `imagePaths_query` to use a different query image.

## How it works
1. **Image encoding:** CLIP's image processor prepares each image, and the image encoder produces a feature vector. The vectors are L2-normalized.
2. **Text encoding:** The tokenizer prepares the text query, and CLIP's text encoder produces a normalized feature vector.
3. **Similarity:** The notebook calculates dot products between the query feature and each collection image feature. Since the features are normalized, these dot products correspond to cosine similarity.
4. **Ranking:** The similarities are multiplied by CLIP's learned logit scale and softmax is applied across the candidate images.

## Notes
- The collection in the example contains only two images; add more images to `imagePaths_data` for a more useful search.
- The softmax values are relative scores across the candidate images, not calibrated real-world probabilities.
- This is an inference/search demo; it does not fine-tune CLIP.
- The model is loaded without an explicit device argument, so this notebook runs on the default device (CPU) unless you move the model and inputs to a GPU.

## Project file
- `search_from_images.ipynb` — notebook implementing text-to-image and image-to-image search.
