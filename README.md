# Image Caption Generator using Deep Learning

This repository contains a deep learning–based **Image Caption Generator** that automatically generates meaningful textual descriptions for images. The project combines techniques from **computer vision** and **natural language processing** to understand image content and express it in natural language.

The project was developed as a **Major Project under the Evoastra AI/ML Internship Program**.

---

## 1. Introduction

Image caption generation is an important problem in artificial intelligence where the goal is to generate a natural language description of an image. This task requires:

- Understanding visual elements present in the image
- Learning relationships between objects
- Converting visual information into grammatically correct sentences

This project addresses the problem using a **CNN–LSTM based encoder–decoder architecture**, which is a widely used and effective approach in multimodal learning.

---

## 2. Problem Statement

Given an input image, the system should automatically generate a descriptive caption that accurately represents the content of the image.

Example:
- Input: Image of a dog playing with a ball
- Output: “A dog is playing with a ball on the grass”

---

## 3. Solution Overview

The project follows a two-stage approach:

1. **Image Feature Extraction**
2. **Caption Generation using Sequence Modeling**

Visual features are extracted from the image using a convolutional neural network. These features are then passed to a recurrent neural network that generates captions word by word.

---

## 4. Model Architecture

### 4.1 Feature Extraction (Encoder)

- Uses **InceptionV3**, a pre-trained convolutional neural network
- Extracts high-level features from images
- Removes the final classification layer to obtain image embeddings

### 4.2 Caption Generation (Decoder)

- Uses **Long Short-Term Memory (LSTM)** networks
- Takes image features and text sequences as input
- Predicts the next word in the caption iteratively
- Uses tokenization and padding for text processing

---

## 5. Dataset

- **MS COCO (Microsoft Common Objects in Context) Dataset**
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
notebooks/: Contains the complete implementation and experimentation code.

models/: Contains trained model files and tokenizer details.

documentation/: Project explanatory documents such as presentation and report.



## 7. Installation and Setup

Step 1: Clone the repository:

git clone https://github.com/Rakshada1811/Image-Caption-Generator-Deep-Learning.git

cd Image-Caption-Generator-Deep-Learning



Step 2: Install required dependencies:

pip install tensorflow keras numpy matplotlib pillow nltk




## 8. Model Files

Due to GitHub file size limitations, trained model files are not fully stored in the repository.

Instructions to download and place the required model files are provided in:

 models/README.md

All downloaded model files must be placed inside the models/ directory before running the notebook.


## 9. How to Run the Project

  **1. Open the Jupyter Notebook:**

jupyter notebook notebooks/image_caption_generator.ipynb

  **2. Ensure all required model files are present in the models/ folder.**

  **3. Run the notebook cells in sequence.**

  **4. Provide an image path as input to generate captions.**



## 10. Results and Observations

- The model generates relevant and context-aware captions
- Captions improve with better image clarity and object representation
- Demonstrates effective integration of CNN and LSTM architectures


## 11. Applications

- Assistive technologies for visually impaired users
- Image indexing and retrieval systems
- Automated content generation
- Human-computer interaction systems

## 12. Author

**Rakshada Renapurkar**
Electronics and Telecomms Graduate
Specialization: Artificial Intelligence and Machine Learning

**Subham Maharana**
**Anurag Ojha**
**K Sai Kiran**
**Kota Aravind Kumar Reddy**
**Nazim Nazir** 


## 13. License

This project is intended for academic and educational purposes.

