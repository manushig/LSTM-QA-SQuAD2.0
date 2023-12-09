import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
from utils.data_utils import normalize_text


class Inference:
    def __init__(self, num_input_tokens, num_output_tokens, input_word_dict,
                 max_input_seq_len, output_word_dict, max_output_len, latent_dim):
        """
        Initializes the Inference class for generating answers using the trained Seq2Seq model.

        Parameters:
            num_input_tokens (int): The number of input tokens in the model's vocabulary.
            num_output_tokens (int): The number of output tokens in the model's vocabulary.
            input_word_dict (dict): Dictionary mapping input words to their indices.
            max_input_seq_len (int): The maximum length of input sequences.
            output_word_dict (dict): Dictionary mapping output words to their indices.
            max_output_len (int): The maximum length of output sequences.
            latent_dim (int): The dimensionality of the latent space in the LSTM layers.
        """
        self.dimensionality = 512

        self.latent_dim = latent_dim
        self.current_script_path = os.path.dirname(os.path.abspath(__file__))
        self.weight_path = os.path.join(self.current_script_path, '..', 'weight')
        self.enc_model_name = "enc_model.h5"
        self.dec_model_name = "dec_model.h5"

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.input_word_dict = input_word_dict
        self.max_input_seq_len = max_input_seq_len
        self.output_word_dict = output_word_dict
        self.max_output_len = max_output_len

        self.enc_path = os.path.join(self.weight_path, self.enc_model_name)
        self.dec_path = os.path.join(self.weight_path, self.dec_model_name)

        # Load models
        self.encoder_model = load_model(self.enc_path)

        self.decoder_model = load_model(self.dec_path)

        self.reverse_output_word_dict = {index: word for word, index in output_word_dict.items()}
        self.data_folder = "../data/train"
        self.tokenizer_path = os.path.join(self.data_folder, 'question_tokenizer.pickle')

        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict(self, input_seq):
        """
        Generates a predicted output sequence for a given input sequence.

        This method uses the trained encoder and decoder models to predict an answer
        based on the input sequence provided. It performs the prediction by encoding
        the input and iteratively decoding to create the output sequence.

        Parameters:
            input_seq (numpy.ndarray): The input sequence to be predicted.

        Returns:
            Tuple[str, float]: A tuple containing the predicted sentence and the time taken for the prediction.
        """
        try:
            start_time = time.time()
            # Encode the input as state vectors.
            states_value = self.encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1 with only the start token.
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = self.output_word_dict['<start>']

            # Sampling loop for a batch of sequences
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

                # Sample a token and add the corresponding word to the decoded sentence
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_word = self.reverse_output_word_dict[sampled_token_index]
                decoded_sentence += ' ' + sampled_word

                # Exit condition: either hit max length or find stop token.
                if sampled_word == '<end>' or len(decoded_sentence) > self.max_output_len:
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index

                # Update states
                states_value = [h, c]
            end_time = time.time()
            response_time = end_time - start_time

            return decoded_sentence, response_time
        except Exception as e:
            print(f"An error occurred: {e}")
            return [], 0

    def preprocess_input(self, raw_input):
        """
        Prepares the raw input text for prediction by normalizing and converting it to a sequence of integers.

        This method applies the same text normalization and tokenization process as used during training. It ensures that
        the input is in the correct format and length for the model to process.

        Parameters:
            raw_input (str): The raw text input provided by the user.

        Returns:
            numpy.ndarray: The preprocessed and padded input sequence ready for prediction.
        """
        # Apply the same normalization as during training
        normalized_input = normalize_text(raw_input)

        # Convert texts to a sequence of integers
        sequence = self.tokenizer.texts_to_sequences([normalized_input])

        # Pad the sequence to the fixed size used during training
        padded_sequence = pad_sequences(sequence, maxlen=self.max_input_seq_len, padding='post')

        return padded_sequence
