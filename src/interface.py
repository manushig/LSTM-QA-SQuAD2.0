import tkinter as tk
from tkinter import Text, END
from tkinter import messagebox
from tkinter import ttk


class QuestionAnsweringBotInterface:
    """
    Class to create a graphical user interface for an AI chatbot.

    Attributes:
        prediction_model: An instance of a class used for predicting responses.
        window (tk.Tk): Main window of the application.
        window_height (int): Height of the main window.
        window_width (int): Width of the main window.
        text_field (tk.Text): Text widget to display chat history.
        input_text_field (tk.Text): Text widget for user input.
    """

    def __init__(self, prediction_model):
        """
        Initializes the ChatInterface with a given prediction model.

        Args:
            prediction_model: An instance of a class used for predicting chatbot responses.
        """
        # Initialize window and layout parameters from config
        self.window = None
        self.window_height = 500
        self.window_width = 1000
        self.text_field_height = int(self.window_height / 20)
        self.text_field_width = int(self.window_width / 10)
        self.input_text_field_height = int(self.text_field_height / 10)
        self.input_text_field_width = self.text_field_width

        # Initialize chat display and input field
        self.text_field = None
        self.input_text = None
        self.input = None
        self.prediction_model = prediction_model

    def launch(self):
        """
        Launches the chat interface window and its components.
        """
        self._initialize_window()
        self._create_chat_display()
        self._create_input_field()
        self.window.mainloop()

    def _initialize_window(self):
        """
        Initializes the main window of the application.
        """
        self.window = tk.Tk()
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.window.minsize(self.window_width, self.window_height)
        self.window.title("QA Bot")

        # Apply a theme to the window for a consistent look
        self.window.tk_setPalette(background='#ECECEC', foreground='#333333')

        # Use ttk Style to change the look of the widgets
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), borderwidth='4')
        style.configure('TLabel', font=('Helvetica', 14), padding=10)
        style.configure('Text', font=('Helvetica', 14), padding=10)

    def _create_chat_display(self):
        """
        Creates the text widget for displaying the chat history.
        """
        self.text_field = Text(self.window, height=self.text_field_height, width=self.text_field_width)
        self.text_field.pack(padx=10, pady=10)

    def _create_input_field(self):
        """
        Creates the user input text widget and the Send button in the chat interface.
        """
        self.input_text_field = Text(self.window, height=self.input_text_field_height,
                                     width=self.input_text_field_width)
        self.input_text_field.pack(padx=10, pady=10)  # Add padding around the input field

        # Bind the Enter key to the process_input function
        self.input_text_field.bind('<Return>', self._process_input)

        send_button = ttk.Button(self.window, text="Send", command=self._process_input)
        send_button.pack(pady=10)

    def _process_input(self, event=None):
        """
        Processes the user input, sends it for prediction, and updates the chat display.
        """
        user_input = self.input_text_field.get("1.0", "end-1c").strip()
        if user_input:
            self.text_field.insert(END, f"User: {user_input}\n")
            # Preprocess the user input
            preprocessed_input = self.prediction_model.preprocess_input(user_input)
            # Get the model's response
            response, time = self.prediction_model.predict(preprocessed_input)
            self.text_field.insert(END, f"Bot: {response}\n\n")
            # Clear input field after processing
            self.input_text_field.delete("1.0", END)

    def _rgb_to_hex(self, rgb):
        """
        Converts RGB values to hexadecimal format.

        Args:
            rgb (tuple): A tuple containing RGB values.

        Returns:
            str: Hexadecimal string.
        """
        return "#%02x%02x%02x" % rgb

    def _on_window_close(self):
        """
        Handles the event when the window close (X) button is clicked.
        """

        # Calculate center position for the application window
        x = (self.window.winfo_screenwidth() // 2) - (self.window_width // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window_height // 2)

        # Reposition the window to the center of the screen
        self.window.geometry(f'+{x}+{y}')

        # Prompt the user with a confirmation dialog
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            # Close the application if the user confirms
            self.window.destroy()
