# Toxic Comment Classification Project

### Project Overview

The aim of this project was to classify almost 160,000 comments into six labels of toxicity. The goal was to develop a model capable of predicting hateful comments on online platforms. Each comment could have zero to six labels indicating different types of toxicity.

- **Model Used**: DistilBERT-cased
- **Labels**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Primary Notebooks**:
    - **EDA**: Exploratory Data Analysis
    - **Model Development and Evaluation**

- [Dataset link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### Technologies

- **Python**
- **PyTorch Lightning**
- **DistilBERT**
- **MLflow**
- **Google Colab**

### Exploratory Data Analysis (EDA)

#### [EDA_Toxic_Comment_Classification Notebook](EDA_Toxic_Comment_Classification.ipynb)

- **Data Insights**:
    
    - Dataset includes 160,000 comments with a 10% toxic and 90% non-toxic split, highlighting class imbalance.
    - Indicators of toxicity included capitalization, symbol use (e.g., exclamation marks), and specific terms like "wikipedia."
    - Lower typo frequency observed in toxic comments.
    - Subjectivity in toxicity labels noted, with discrepancies in perceived toxicity levels.
    - No missing values or duplicates, ensuring data integrity.
    - Comment lengths varied from 1 to 1500 words, averaging around 67 words.
    - Challenges in identifying non-English comments.
- **EDA Highlights**:
    
    - Addressed missing values and duplicates.
    - Focused on label distributions, word frequencies, correlations, comment lengths, symbol usage, and capitalization ratios.
    - Visualized the findings and provided insights for potential improvements.

### Model Development and Evaluation

#### [Training_and_eval Notebook](training_and_eval.ipynb)

- **Model Implementation**:
    
    - Tokenization for preprocessing.
    - Training with PyTorch Lightning and DistilBERT-cased.
    - Visualization of model metrics using MLflow.
    - Computations performed on Google Colab.
    - Evaluation on test data, merging labels and comments, and removing "-1" values from the test dataset labels.
    - Discussion of issues and potential improvements.
    - Model parameters based on various sources: blogs, official documentation, and tutorials.
- **Challenges and Learnings**:
    
    - Implemented early stopping with patience = 3.
    - Added class weights, though results were underwhelming.
    - Tracking validation and training losses with MLflow.
    - Issues with deciding how many epochs to keep the backbone frozen and understanding sudden validation loss spikes.
    - Recognized the need for better handling of certain elements in the model architecture for improved performance.

### License

This project is made available under the MIT License, supporting open-source collaboration and knowledge sharing.
