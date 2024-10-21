# Sentiment Analysis Using BERT

## Overview
This project utilizes **BERT** (Bidirectional Encoder Representations from Transformers) for sentiment analysis on climate change-related data, leveraging the **Hugging Face Transformers** library. BERT is a powerful model for natural language processing (NLP) that captures deep, bidirectional context, making it effective for classification tasks. In this project, we use **DistilBERT**, a smaller and faster version of BERT, to classify the sentiment of tweets about climate change.

## Dataset
The dataset for this project is the **Sentiment of Climate Change** dataset, available on [data.world](https://data.world/crowdflower/sentiment-of-climate-change). It contains tweets related to climate change, labeled by sentiment.

## Features
- **Tokenizer and Model**: We use the DistilBERT model from the Hugging Face library to tokenize the text and generate sentence embeddings.
- **Logistic Regression**: After extracting features using BERT, a logistic regression classifier is trained for sentiment analysis.
- **Google Colab**: Given the high computational requirements of BERT, Google Colab was used to train and run the model.

## Project Workflow

1. **Import Required Libraries**:  
   Libraries like **Pandas**, **Scikit-learn**, **NumPy**, **Transformers**, and **Torch** are used throughout the project.

2. **Data Loading and Preprocessing**:
   - The dataset is loaded as a CSV file from data.world.
   - Due to memory limitations, only the first 2000 records are used.
   - The labels in the 'existence' column were cleaned to ensure consistency.

3. **Tokenization and Padding**:
   - The **BERT Tokenizer** converts the input text into tokens.
   - Sentences are padded with zeros to ensure they have the same length.

4. **Attention Mask Creation**:
   - An **attention mask** is used to distinguish real tokens from padded ones.

5. **Feature Extraction**:
   - The tokenized and padded inputs are fed into **DistilBERT**, which outputs the last hidden states used as features for the classifier.

6. **Model Training**:
   - A **Logistic Regression** model is trained using the features extracted from BERT.

7. **Prediction**:
   - After training, the model is used to predict the sentiment of a sample review.

## Installation Instructions

To run the project, install the required dependencies:

```sh
pip install transformers
```

Other necessary libraries include **Pandas**, **NumPy**, **Scikit-learn**, and **Torch**. Ensure they are installed before proceeding.

## How to Run the Project
1. Clone the repository to your local system:
   ```sh
   git clone <repository-url>
   ```

2. Install the dependencies listed above.

3. Run the notebook or Python script to see the full sentiment analysis workflow.

4. Replace the sample text in the prediction section to test custom inputs.

## Steps in Sentiment Analysis Using BERT

1. **Install Transformers Library**:  
   Install the transformers library with:
   ```sh
   !pip install transformers
   ```

2. **Initialize Model and Tokenizer**:  
   Use **DistilBERT** as a lightweight version of BERT.

3. **Tokenize Input**:  
   Convert input text into tokens using the BERT tokenizer.

4. **Padding**:  
   Pad all tokenized texts to ensure equal length.

5. **Attention Mask Preparation**:  
   Create an attention mask where all real tokens are `1` and padded tokens are `0`.

6. **Conversion to Torch Tensors**:  
   Convert padded input and attention mask into **torch tensors**.

7. **Feature Extraction Using BERT**:  
   Pass tensors through BERT to obtain the last hidden states and extract features.

8. **Train Model**:  
   Use **Logistic Regression** to classify the extracted features.

9. **Predict Sentiment**:  
   Use the trained model to predict sentiment for new sample texts.

## Requirements

- Python 3.6+
- PyTorch
- Transformers Library
- Google Colab (or another GPU-enabled environment)
- Scikit-learn

## Notes

- We used **DistilBERT** to save computational resources, but other versions of BERT can also be used by modifying the tokenizer and model initialization.
- Due to hardware limitations, only the first 2000 records were processed. If you have more resources, feel free to use a larger dataset.

## Future Improvements
- **Improve Model**: Experiment with more sophisticated classifiers, such as **SVM** or **Deep Neural Networks**, to enhance performance.
- **Data Augmentation**: Increase the diversity of the dataset with data augmentation techniques to improve robustness.
- **Fine-Tuning**: Fine-tune BERT specifically for this dataset to achieve better accuracy.

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [Sentiment of Climate Change Dataset on data.world](https://data.world/crowdflower/sentiment-of-climate-change)

## License
This project is licensed under the MIT License.

## Contact
For more information or questions, feel free to reach out:
- https://www.linkedin.com/in/anandlo/
- Email: anandlo@dal.ca

