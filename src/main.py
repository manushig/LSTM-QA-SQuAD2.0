from interface import QuestionAnsweringBotInterface
from preprocessing import DataPreprocessor
from seq2seq import Training
from inference import Inference
import matplotlib.pyplot as plt
import pandas as pd


def main():
    """
    Main function to run the data preprocessing.
    It initializes the DataPreprocessor, executes preprocessing,
    retrieves training parameters.
    """
    try:
        # Data preprocessing
        preprocessor = DataPreprocessor()
        preprocessor.execute_preprocessing()
        num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data = \
            preprocessor.get_training_parameters()

        # Model training and cross-validation
        training = Training()

        avg_accuracy, avg_loss = training.cross_validate(num_input_tokens, num_output_tokens,
                                                         encoder_input_data, decoder_input_data,
                                                         decoder_target_data, k=5)

        encoder_model, decoder_model = training.train_model(num_input_tokens, num_output_tokens,
                                                            encoder_input_data, decoder_input_data,
                                                            decoder_target_data)

        # Model inference setup
        input_word_dict, max_input_seq_len, output_word_dict, max_output_len = preprocessor.get_test_parameters()
        response_predictor = Inference(num_input_tokens, num_output_tokens, input_word_dict,
                                       max_input_seq_len, output_word_dict, max_output_len, latent_dim=128)

        # User interface launch
        interface = QuestionAnsweringBotInterface(response_predictor)
        interface.launch()
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], 0


def plot_cross_validation_results(file_path):
    """
    Plots cross-validation results for model accuracy and loss.

    Args:
        file_path (str): Path to the CSV file containing cross-validation results.
    """
    df = pd.read_csv(file_path)
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Fold'], df['Accuracy'], marker='o')
    plt.title('Cross-Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Fold'], df['Loss'], marker='o', color='red')
    plt.title('Cross-Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('../image/crossvalidation.png')
    plt.show()


if __name__ == "__main__":
    main()
    plot_cross_validation_results('../data/cross_validation_results.csv')
