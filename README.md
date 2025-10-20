# Urdu Transformer Chatbot 🤖

A GPT-style chatbot for Urdu text, built using a custom Transformer model implemented in PyTorch. It can generate responses in Urdu based on input sentences using greedy or beam search decoding.

---

## Features

- Transformer-based architecture (Encoder-Decoder with Multi-Head Attention)  
- Custom tokenization and normalization for Urdu text  
- Greedy and beam search text generation  
- Streamlit-based interactive web UI  
- Trained on ~20,000 Urdu sentences  

---

## Demo

You can run the chatbot locally with Streamlit:

```bash
streamlit run app.py
```
Type your message in Urdu, and the chatbot will respond in Urdu.

## Installation

Clone this repository:
```bash
git clone https://github.com/rafayfarooq768/urdu-chatbot.git
cd urdu-chatbot
```
Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
Install required packages:
```bash
pip install -r requirements.txt
```
## Project Structure
```bash
urdu-chatbot/
│
├─ app.py               # Streamlit web app
├─ model.py             # Transformer model implementation
├─ tokenizer.pkl        # Saved tokenizer
├─ transformer_urdu_model.pth  # Trained model weights
├─ README.md
└─ requirements.txt
```
## How to Use
Load the Streamlit app:
```bash
streamlit run app.py
```
Enter your Urdu sentence in the input box.
The chatbot will generate a response in Urdu.

## Training
The model was trained using:
Data: ~20,000 Urdu sentences
Architecture: 2-layer Transformer, 128 embedding, 4 attention heads, 512 feed-forward dimension
Loss: CrossEntropyLoss ignoring padding tokens
Optimizer: Adam (learning rate 1e-4)
Epochs: 30

## Notes
The model requires transformer_urdu_model.pth to generate text.

## License
This project is open-source under the MIT License.

## Acknowledgements
Urdu dataset: muhammadahmedansari/urdu-dataset-20000
Inspired by PyTorch Transformer tutorials and Hugging Face concepts.
