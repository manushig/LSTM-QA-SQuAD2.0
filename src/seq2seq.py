import os
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, AdditiveAttention
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


class Training:
    """
    Manages the training process of a Seq2Seq model for a question-answering system.

    Attributes:
        dimensionality (int): Dimensionality of the LSTM layers.
        batch_size (int): Batch size for training the model.
        epochs (int): Number of epochs for training.
        (Other paths and model configuration attributes...)
    """

    def __init__(self):
        """
        Initializes the Training class with default values for model training, paths for storing models, and plotting.

        Sets up paths for images, weights, data, and model file names. Initializes hyperparameters like embedding dimensions and latent dimensions.
        Also, checks for available GPU resources.
        """
        self.dimensionality = 512
        self.batch_size = 32
        self.epochs = 100

        self.current_script_path = os.path.dirname(os.path.abspath(__file__))
        self.image_path = os.path.join(self.current_script_path, '..', 'image')
        self.weight_path = os.path.join(self.current_script_path, '..', 'weight')
        self.data_path = os.path.join(self.current_script_path, '..', 'data')

        self.model_name = "model.h5"
        self.encoder_model_name = "enc_model.h5"
        self.decoder_model_name = "dec_model.h5"
        self.plot_name = "acc.png"

        self.model_path = os.path.join(self.weight_path, self.model_name)
        self.encoder_model_path = os.path.join(self.weight_path, self.encoder_model_name)
        self.decoder_model_path = os.path.join(self.weight_path, self.decoder_model_name)
        self.acc_image_path = os.path.join(self.image_path, self.plot_name)

        self.emb_dim = 256
        self.latent_dim = 128
        print("Available GPU's:", self.get_available_gpus())

    def train_model(self, num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data):
        """
        Trains the Seq2Seq model and saves the trained models.

        If the model has not been trained before, it trains the model with the given data and saves the history.
        If the model already exists, it loads the model. After training or loading, it extracts and saves the encoder and decoder models.

        Parameters:
            num_input_tokens (int): Number of input tokens for the model.
            num_output_tokens (int): Number of output tokens for the model.
            encoder_input_data (numpy.ndarray): Encoder input data.
            decoder_input_data (numpy.ndarray): Decoder input data.
            decoder_target_data (numpy.ndarray): Decoder target data for training.

        Returns:
            Tuple[Model, Model]: The trained encoder and decoder models.
        """
        self.model = self.create_model(num_input_tokens, num_output_tokens)
        # Encoder

        if not os.path.exists(self.model_path):
            history = self.model.fit(
                [encoder_input_data, decoder_input_data], decoder_target_data,
                validation_split=0.2, batch_size=self.batch_size, epochs=self.epochs, shuffle=True,
                callbacks=[EarlyStopping(monitor='val_loss', patience=15),
                           ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]
            )
            self.save_history(history)
            self.model.save(self.model_path)
        else:
            self.model = load_model(self.model_path)

        # Save the final encoder and decoder models
        encoder_model, decoder_model = self.extract_encoder_decoder()
        encoder_model.save(self.encoder_model_path)
        decoder_model.save(self.decoder_model_path)

        return encoder_model, decoder_model

    def extract_encoder_decoder(self):
        """
        Extracts and returns the encoder and decoder models from the trained Seq2Seq model.

        This method separates the encoder and decoder parts from the full Seq2Seq model,
        allowing for their independent use, typically during the inference phase.

        Returns:
            Tuple[Model, Model]: The extracted encoder and decoder models.
        """
        # Extracting the encoder
        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[4].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        # Extracting the decoder
        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm = self.model.layers[5]
        dec_emb_layer = self.model.layers[3]
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def save_history(self, history):
        """
        Plots and saves the training and validation accuracy and loss graphs.

        Parameters:
            history (History): A Keras History object containing the training/validation loss and accuracy.
        """
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.savefig(self.acc_image_path)
        plt.show()

    @staticmethod
    def get_available_gpus():
        """
        Lists the available GPU devices.

        This method checks and returns a list of GPU devices available on the system.
        Useful for ensuring that TensorFlow can access the GPUs for accelerated computation.

        Returns:
            List[str]: A list of available GPU device names.
        """
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def cross_validate(self, num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data,
                       decoder_target_data, k=5):
        """
        Performs k-fold cross-validation on the Seq2Seq model.

        Trains the model multiple times (k times) on different subsets of the data
        and evaluates performance to provide a more robust assessment of the model's efficacy.

        Parameters:
            num_input_tokens (int): Number of unique input tokens.
            num_output_tokens (int): Number of unique output tokens.
            encoder_input_data (numpy.ndarray): Encoded input data for the encoder.
            decoder_input_data (numpy.ndarray): Encoded input data for the decoder.
            decoder_target_data (numpy.ndarray): Target data for the decoder.
            k (int): Number of folds for cross-validation.

        Returns:
            Tuple[float, float]: Average accuracy and loss across all folds.
        """
        # Check if the model already exists
        if os.path.exists(self.model_path):
            return None, None

        kf = KFold(n_splits=k, shuffle=True)
        accuracies = []
        losses = []

        for train_index, val_index in kf.split(encoder_input_data):
            # Create a new model instance for each fold
            self.model = self.create_model(num_input_tokens, num_output_tokens)

            train_encoder_input, val_encoder_input = encoder_input_data[train_index], encoder_input_data[val_index]
            train_decoder_input, val_decoder_input = decoder_input_data[train_index], decoder_input_data[val_index]
            train_decoder_target, val_decoder_target = decoder_target_data[train_index], decoder_target_data[val_index]

            # Train the model
            self.model.fit([train_encoder_input, train_decoder_input], train_decoder_target,
                           validation_data=([val_encoder_input, val_decoder_input], val_decoder_target),
                           batch_size=self.batch_size, epochs=self.epochs, shuffle=True,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=15),
                                      ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)])

            # Evaluate the model
            loss, accuracy = self.model.evaluate([val_encoder_input, val_decoder_input], val_decoder_target)
            accuracies.append(accuracy)
            losses.append(loss)

        # Storing results in a DataFrame
        results_df = pd.DataFrame({
            'Fold': range(1, k + 1),
            'Accuracy': accuracies,
            'Loss': losses
        })

        # Calculate and add averages
        avg_accuracy = np.mean(accuracies)
        avg_loss = np.mean(losses)
        avg_row = pd.DataFrame({'Fold': ['Average'], 'Accuracy': [avg_accuracy], 'Loss': [avg_loss]})

        results_df = pd.concat([results_df, avg_row], ignore_index=True)
        # Save to CSV
        results_df.to_csv(os.path.join(self.data_path, 'cross_validation_results.csv'), index=False)

        # return the averages for further processing
        return avg_accuracy, avg_loss

    def create_model(self, num_input_tokens, num_output_tokens):
        """
        Creates and compiles a new instance of the Seq2Seq model.

        This method sets up the architecture for the Seq2Seq model, including the encoder and decoder components,
        and compiles the model with appropriate optimizer and loss function.

        Parameters:
            num_input_tokens (int): The size of the input vocabulary.
            num_output_tokens (int): The size of the output vocabulary.

        Returns:
            Model: The compiled Seq2Seq model.
        """
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(num_input_tokens, self.emb_dim, embeddings_regularizer=l2(1e-4))(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, dropout=0.2, recurrent_dropout=0.2,
                            kernel_constraint=max_norm(3))
        _, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(num_output_tokens, self.emb_dim)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.2,
                            recurrent_dropout=0.2, kernel_constraint=max_norm(3))
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

        # Dense layer for prediction
        decoder_dense = TimeDistributed(Dense(num_output_tokens, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define and compile the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
