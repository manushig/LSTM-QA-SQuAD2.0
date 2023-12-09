# LSTM-Based Question Answering System Leveraging SQuAD 2.0

## Overview
This project develops an advanced LSTM-based Question Answering system using the SQuAD 2.0 dataset. It features a comprehensive approach encompassing data preprocessing, LSTM Seq2Seq model implementation, rigorous evaluation, and an intuitive GUI for user interaction.

## Features
- Employs the complex SQuAD 2.0 dataset, including unanswerable questions for a realistic challenge.
- Implements an LSTM Seq2Seq model with attention mechanisms for accurate response generation.
- Includes meticulous data preprocessing and model training, achieving high accuracy and validation scores.
- Offers a user-friendly Tkinter-based interface for seamless interaction.

## Prerequisites
Before running the code, ensure you have the following installed:
- Python 3.9.0
- Tensorflow 2.14.0
- CUDA 11.2
- PANDAS 2.1.3
- NUMPY 1.26.2
- DATASETS 2.14.7
- CUDNN 11.4 (necessary only for gpu computing (optional))
  
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/manushig/LSTM-QA-SQuAD2.0.git
   ```
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the system, follow these steps:

1. Run main.py to preprocess data, train, test, and interact with the model.
   
   ```python
   python main.py
   ```
   
## Execution Process
To run the system, execute `main.py` which encompasses the following stages:

1. **Data Preprocessing**: The script begins with data preprocessing using `preprocessing.py`. This step includes cleaning and structuring the dataset for model training.

2. **Model Training and Cross-Validation**: Next, the LSTM Seq2Seq model defined in `seq2seq.py` is trained. It includes a cross-validation process to ensure model robustness and accuracy. This stage involves the learning process where the model adapts to answer questions based on the SQuAD 2.0 dataset.

3. **Model Evaluation**: The model is then evaluated for its performance. Testing and validation are done to ensure the accuracy and reliability of the model.

4. **User Interface Launch**: Finally, the script launches a user interface developed in `interface.py`. This GUI allows users to interact with the question-answering system in a user-friendly manner.

## Data
The script processes the SQuAD v2.0 dataset, which can be downloaded automatically by the script or manually from [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/).

## Components
- `main.py`: Orchestrates the workflow, including data preprocessing, model training, evaluation, and launching the user interface.
- `preprocessing.py`: Handles data preprocessing, including cleaning, tokenization, and preparing the SQuAD 2.0 dataset for the LSTM model.
- `seq2seq.py`: Defines the LSTM Seq2Seq model architecture, incorporating attention mechanisms for improved performance.
- `inference.py`: Facilitates model inference, generating answers to user-posed questions based on the trained model.
- `interface.py`: Provides a graphical user interface (GUI) using Tkinter for easy interaction with the question-answering system.

## Contact
For inquiries or contributions, feel free to reach out to any of our team members:

- [Manushi](manushi.f@northeastern.edu)
- [Ameya Santosh Gidh](gidh.am@northeastern.edu)
- [Narayana Sudheer Vishal Basutkar](basutkar.n@northeastern.edu)

Project Link: [GitHub](https://github.com/manushig/LSTM-QA-SQuAD2.0)
