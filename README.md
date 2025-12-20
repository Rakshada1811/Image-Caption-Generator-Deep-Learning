# Image Caption Generator using Deep Learning

This repository contains a deep learning–based Image Caption Generator that automatically generates meaningful textual descriptions for images. The project combines techniques from computer vision and natural language processing to understand image content and express it in natural language.

The project was developed as a Major Project under the Evoastra AI/ML Internship Program.

---

## 1. Introduction

Image caption generation is an important problem in artificial intelligence where the goal is to generate a natural language description of an image. This task requires:

- Understanding visual elements present in the image  
- Learning relationships between objects  
- Converting visual information into grammatically correct sentences  

This project addresses the problem using a CNN–LSTM based encoder–decoder architecture, which is a widely used and effective approach in multimodal learning.

---

## 2. Problem Statement

Given an input image, the system should automatically generate a descriptive caption that accurately represents the content of the image.

Example:  
- Input: Image of a dog playing with a ball  
- Output: “A dog is playing with a ball on the grass”

---

## 3. Solution Overview

The project follows a two-stage approach:

1. Image Feature Extraction  
2. Caption Generation using Sequence Modeling  

Visual features are extracted from the image using a convolutional neural network. These features are then passed to a recurrent neural network that generates captions word by word.

---

## 4. Model Architecture

### 4.1 Feature Extraction (Encoder)

- Uses InceptionV3, a pre-trained convolutional neural network  
- Extracts high-level features from images  
- Removes the final classification layer to obtain image embeddings  

### 4.2 Caption Generation (Decoder)

- Uses Long Short-Term Memory (LSTM) networks  
- Takes image features and text sequences as input  
- Predicts the next word in the caption iteratively  
- Uses tokenization and padding for text processing  

---

## 5. Dataset

- MS COCO (Microsoft Common Objects in Context) Dataset  
- Contains thousands of images with multiple human-written captions  
- Used for training and evaluating the caption generation model  

---

## 6. Repository Structure

```text
Image-Caption-Generator-Deep-Learning/
│
├── notebooks/
│   ├── image_caption_generator.ipynb
│   └── README.md
│
├── models/
│   ├── tokenizer.pkl
│   └── README.md
│
├── documentation/
│   └── README.md
│
├── presentation/
│   └── Project_Presentation.pdf
│
└── README.md

```
### Folder Description

- `notebooks/` – Contains the complete Jupyter Notebook used for data preprocessing, model development, training, and testing  
- `models/` – Stores the trained model weights and tokenizer generated after training  
- `documentation/` – Includes supporting documentation explaining the project design and workflow  
- `presentation/` – Contains the final project presentation used for evaluation  

---

## 7. Project Methodology

The project was developed using a structured deep learning workflow that integrates computer vision and natural language processing. The overall methodology followed is outlined below.

---

## 8. Data Preparation

- The MS COCO dataset was used as the primary data source  
- Images were resized and normalized to match the input requirements of the CNN  
- Captions were cleaned by removing punctuation and converting text to lowercase  
- Special start and end tokens were added to each caption  
- Tokenization and padding were applied to convert text into numerical sequences  

---

## 9. Feature Extraction

- A pre-trained InceptionV3 model was used for extracting visual features  
- The final classification layer was removed  
- Image embeddings were generated and stored for efficient training  
- This step helped reduce computational cost during caption generation  

---

## 10. Caption Generation Model

- An LSTM-based neural network was used to model the sequential nature of language  
- Image features were combined with embedded caption sequences  
- The model was trained to predict the next word given the image context and previous words  
- Categorical cross-entropy loss and Adam optimizer were used during training  

---

## 11. Model Training and Evaluation

- The model was trained on paired image–caption data  
- Performance was evaluated by observing the quality and coherence of generated captions  
- Greedy decoding was used during inference to generate final captions  

---

## 12. Results and Observations

- The model successfully generates meaningful captions for unseen images  
- Captions reflect object presence and contextual relationships  
- Performance improves with clearer images and well-represented objects  

---

## 13. Applications

- Image description systems for visually impaired users  
- Automated image tagging and indexing  
- Multimedia content analysis  
- Intelligent human–computer interaction systems  

---

## 14. Authors

Rakshada Renapurkar  
Electronics and Telecommunication Graduate  
Specialization: Artificial Intelligence and Machine Learning  

Subham Maharana  
Anurag Ojha  
K Sai Kiran  
Kota Aravind Kumar Reddy  
Nazim Nazir  

---

## 15. License

This project is intended for academic and educational purposes.
