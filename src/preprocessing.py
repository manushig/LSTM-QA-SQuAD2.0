import ast
import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.data_utils import normalize_text


class DataPreprocessor:
    def __init__(self, data_limit: int = 10000, data_folder: str = "../data/train",
                 data_name: str = "SquadDataset.csv"):
        """
        Initializes the DataPreprocessor class with specified parameters.

        Parameters:
        data_limit (int): Limit for the number of data points to process.
        data_folder (str): Directory where data files are stored.
        data_name (str): Name of the dataset file.
        """
        self.data_limit = data_limit
        self.data_folder = data_folder
        self.data_name = data_name
        self.data_final_name = "SquadDatasetFinal.csv"
        self.data_path = os.path.join(self.data_folder, self.data_name)
        self.final_data_path = os.path.join(self.data_folder, self.data_final_name)
        self.data_frame = None
        self.num_input_tokens = None
        self.num_output_tokens = None
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        self.input_word_dict = None
        self.max_input_seq_len = None
        self.output_word_dict = None
        self.max_output_len = None

    def execute_preprocessing(self):
        """
        Executes the entire preprocessing pipeline.
        """
        try:
            if os.path.exists(self.final_data_path):
                self.data_frame = pd.read_csv(self.final_data_path)
            else:
                self._load_or_fetch_data()
            self._process_data()
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

    def _load_or_fetch_data(self):
        """
        Loads data from local storage or fetches it from the SQuAD dataset if not available.
        """
        try:
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder, exist_ok=True)
            if not os.path.exists(self.data_path):
                self._fetch_squad_dataset()
            self.data_frame = pd.read_csv(self.data_path)
            self._handle_missing_and_extract_answers()
            self._preprocess_text_columns()
            self.data_frame.to_csv(self.final_data_path, index=False)
        except Exception as e:
            print(f"Error loading or fetching data: {e}")
            raise

    def _fetch_squad_dataset(self):
        """
        Fetches the SQuAD dataset and saves it locally.
        """
        try:
            dataset = load_dataset('squad_v2', split='train')
            self.data_frame = pd.DataFrame(data=dataset)
            self.data_frame.to_csv(self.data_path)
        except Exception as e:
            print(f"Error fetching SQuAD dataset: {e}")
            raise

    def _extract_answer_text(self, answer_dict_str: str) -> str:
        """
        Extracts text from the answer dictionary string.

        Parameters:
        answer_dict_str (str): A string representation of the answer dictionary.

        Returns:
        str: Extracted text from the answer dictionary.
        """
        try:
            answer_dict = ast.literal_eval(answer_dict_str)
            text_list = answer_dict.get("text", [])
            return text_list[0] if text_list else ""
        except Exception as e:
            print(f"Error extracting answer text: {e}")
            raise

    def _handle_missing_and_extract_answers(self):
        """
        Handles missing data and extracts answer texts from the 'answers' column.
        """
        try:
            self.data_frame.dropna(axis=0, inplace=True)
            self.data_frame["answers"] = self.data_frame["answers"].apply(self._extract_answer_text)
            self.data_frame['answers'] = self.data_frame['answers'].apply(
                lambda x: x if x.strip() != "" else "<no_answer>")
        except Exception as e:
            print(f"Error handling missing data or extracting answers: {e}")
            raise

    def _preprocess_text_columns(self):
        """
        Preprocesses the main text columns ('question' and 'answers') and saves the processed data.
        """
        try:
            self.data_frame = self.data_frame[["question", "answers"]]
            self.data_frame["question"] = self.data_frame["question"].apply(normalize_text)
            self.data_frame["answers"] = self.data_frame["answers"].apply(
                lambda x: f'<start> {normalize_text(x)} <end>')
            self.data_frame = self.data_frame.iloc[:self.data_limit]
            self.data_frame.to_csv(self.final_data_path)
        except Exception as e:
            print(f"Error preprocessing text columns: {e}")
            raise

    def _tokenization_and_encoding(self, texts: List[str]) -> Tuple[Tokenizer, List[List[int]]]:
        """
                Tokenizes and encodes the given texts.

                Parameters:
                texts (List[str]): A list of text strings to tokenize and encode.

                Returns:
                Tuple[Tokenizer, List[List[int]]]: A tuple containing the tokenizer and list of encoded sequences.
                """
        try:
            tokenizer = Tokenizer(oov_token='<oov>', filters='')  # Modify filters
            tokenizer.fit_on_texts(texts)

            # Manually add <start> and <end> if not already included
            if '<start>' not in tokenizer.word_index:
                tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
            if '<end>' not in tokenizer.word_index:
                tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

            sequences = tokenizer.texts_to_sequences(texts)
            return tokenizer, sequences
        except Exception as e:
            print(f"Tokenization and encoding failed: {e}")
            raise

    def _pad_sequences(self, sequences: List[List[int]], maxlen: int) -> List[List[int]]:
        """
        Pads the given sequences to a specified maximum length.

        Parameters:
        sequences (List[List[int]]): Sequences to pad.
        maxlen (int): Maximum length of sequences after padding.

        Returns:
        List[List[int]]: Padded sequences.
        """
        try:
            return pad_sequences(sequences, padding='post', truncating='post', maxlen=maxlen)
        except Exception as e:
            print(f"Padding sequences failed: {e}")
            raise

    def _save_tokenizer(self, tokenizer, filename):
        """
        Saves the tokenizer to a file.

        Parameters:
        tokenizer (Tokenizer): The tokenizer to save.
        filename (str): The filename to save the tokenizer to.
        """
        tokenizer_path = os.path.join(self.data_folder, filename)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _process_data(self):
        """
        Processes the data including tokenization, encoding, and sequence padding.
        Also calculates and logs performance metrics.
        """
        try:
            questions, answers = list(self.data_frame['question']), list(self.data_frame['answers'])

            question_tokenizer, encoded_questions = self._tokenization_and_encoding(questions)
            answer_tokenizer, encoded_answers = self._tokenization_and_encoding(answers)

            self._save_tokenizer(question_tokenizer, 'question_tokenizer.pickle')
            self._save_tokenizer(answer_tokenizer, 'answer_tokenizer.pickle')

            self.max_input_seq_len = max(len(seq) for seq in encoded_questions)
            self.max_output_len = max(len(seq) for seq in encoded_answers)

            self.num_input_tokens = len(question_tokenizer.word_index) + 1
            self.num_output_tokens = len(answer_tokenizer.word_index) + 1
            self.input_word_dict = question_tokenizer.word_index
            self.output_word_dict = answer_tokenizer.word_index

            total_tokens = sum([len(seq) for seq in encoded_questions + encoded_answers])
            vocab_size = len(set(question_tokenizer.word_index.values()) |
                             set(answer_tokenizer.word_index.values()))

            print(f"Total tokens: {total_tokens}")
            print(f"Vocabulary size: {vocab_size}")
            print(f"Max input sequence length: {self.max_input_seq_len}")
            print(f"Max output sequence length: {self.max_output_len}")

            self.encoder_input_data = self._pad_sequences(encoded_questions, self.max_input_seq_len)
            self.decoder_input_data = self._pad_sequences(encoded_answers, self.max_output_len)

            offset_encoded_answers = [seq[1:] for seq in encoded_answers]
            self.decoder_target_data = self._pad_sequences(offset_encoded_answers, self.max_output_len)

        except Exception as e:
            print(f"Processing data failed: {e}")
            raise

    def get_training_parameters(self) -> Tuple[int, int, List[List[int]], List[List[int]], List[List[int]]]:
        """
        Retrieves the training parameters.

        Returns:
        Tuple[int, int, List[List[int]], List[List[int]], List[List[int]]]: Training parameters including
        number of input tokens, number of output tokens, encoder input data, decoder input data, and decoder target data.
        """
        if (self.num_input_tokens is None or self.num_output_tokens is None or
                self.encoder_input_data is None or np.size(self.encoder_input_data) == 0 or
                self.decoder_input_data is None or np.size(self.decoder_input_data) == 0 or
                self.decoder_target_data is None or np.size(self.decoder_target_data) == 0):
            print("Training parameters requested before preprocessing has been completed.")
            raise ValueError("Preprocessing must be completed before getting training parameters.")
        return (self.num_input_tokens, self.num_output_tokens,
                self.encoder_input_data, self.decoder_input_data, self.decoder_target_data)

    def get_test_parameters(self) -> Tuple[Dict[str, int], int, Dict[str, int], int]:
        """
        Retrieves the test parameters.

        Returns:
        Tuple[Dict[str, int], int, Dict[str, int], int]: Test parameters including
        input word dictionary, maximum input sequence length, output word dictionary, and maximum output length.
        """
        if not all([self.input_word_dict, self.max_input_seq_len, self.output_word_dict, self.max_output_len]):
            print("Test parameters requested before preprocessing has been completed.")
            raise ValueError("Preprocessing must be completed before getting test parameters.")
        return self.input_word_dict, self.max_input_seq_len, self.output_word_dict, self.max_output_len
