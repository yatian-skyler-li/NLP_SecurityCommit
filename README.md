# Security Analysis through Commit History

## Project Overview
This project involves using machine learning (ML) and natural language processing (NLP) to facilitate an automated analysis of commit logs for enhanced security analysis.

## Getting Started

### Prerequisites

* various libraries were used in this project. 
* csv, pandas, and chardet were used to load the file and analyze the data type in the field of the file
* nltk, transformers were used for data preprocessing and tokenization
* torch, doc2vec, sklearn were used for ML models. For example, SVM, LogisticRegression etc.

## Usage

* Models used for binary classification: SVM, RNN, RandomForest, linearRegression, BERT, CODEBERT, GRAPHCODEBERT
* Models used for multiclass classification: SVM, NaiveBayes, logisticRegression

## Methodology

The project follows these key steps:

1. **Extract commit logs**: Combine the negative message file and postive (security related) message files and add binary labels. 
For multiclass classification, we only focus positive messages and the classification is base on CEWID.

2. **Preprocess commit logs**: Tokenizing the loaded messages

3. **Apply ML models**: Machine learning models were chosen and trained. See the comments in the scripts for details

## Results and Evaluation

* Generally, we use 80% of the data for training, and 20% for testing in binary classification
* In multiclass classification, we focus on the CWEIDs with a frequency of over 400. This way, we provide enough data to train the models.
* Calculate the f1, precision, and accuracy of the model. 

## Authors and Acknowledgment

* Yatian Li - Initial work - yatian.li@uq.edu.au
* UQ AI internship project, supervised by Dr. Guowei Yang, School of ITEE, The University of Queensland
