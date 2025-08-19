This project is a Transformer-based sentiment classifier built for analyzing airline-related tweets. It predicts whether a tweet is positive, negative, or neutral using Hugging Face Transformers.

Airline-Sentiment-Classifier/
│── app.py                        # Main application script
│── Airline_Model.ipynb            # Jupyter Notebook (training + testing)
│── AirlineTweets.csv / .xlsx      # Dataset of airline tweets
│── fine-tuned-airline-model/      # (Not uploaded, too large for GitHub)
│── README.md                      # Documentation



⚡ Features
Uses a pre-trained Transformer model fine-tuned for sentiment analysis.
Supports training and inference via Jupyter Notebook (Airline_Model.ipynb).
app.py script for running predictions on new text data.


🚀 How to Run
Since the trained model is very large (~255 MB), it is not uploaded to GitHub.
However, you can still run the notebook and app by following these steps:

1️⃣ Clone the repository
git clone https://github.com/your-username/Airline-Sentiment-Classifier.git
cd Airline-Sentiment-Classifier

2️⃣ Install dependencies
Create a virtual environment (recommended), then install requirements:
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook
Open the notebook and execute all cells:
jupyter notebook Airline_Model.ipynb
This will automatically download the required Hugging Face model weights.
The model will be cached locally (~/.cache/huggingface/transformers).

4️⃣ Run the App Script
Once the model is available locally, you can run the Python app:
python app.py


📌 Notes
The fine-tuned model folder is not included in this repository due to GitHub’s file size limits.
Hugging Face will automatically download the required model when you run the notebook.
If you want to share the trained model with others, you can upload it to Hugging Face Hub or Google Drive, then update your code to download from there.
