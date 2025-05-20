##🧠 Turkish Legal QA System

This repository contains a custom-trained Transformer-based Question Answering (QA) system built for Turkish legal texts. The system is capable of training on a custom tokenizer and dataset, and post-processing predictions using a large language model (LLaMA 3 via Groq API) and translation to provide clearer Turkish responses.
#📁 Files Overview
model_training.py

This script handles training a Transformer-based QA model using PyTorch.
🔍 Key Components:

- TransformerQA: A lightweight Transformer encoder with learned positional embeddings and a final classification head for start and end token prediction.

- QADataset: A PyTorch Dataset that prepares question-context-answer triples using a PreTrainedTokenizerFast tokenizer.

- Tokenizer: Loaded from a local JSON file (trained_tokenizer.json) and padded if necessary.

- Training Loop: Implements loss computation with CrossEntropyLoss, optimizer with Adam, and early stopping based on validation loss.

- Evaluation: F1 score computation is used as a secondary metric.

- Model Saving: Best-performing model is saved to transformer_qa_model.pth.

post_processing_model.py

This script loads the trained model and performs prediction, followed by response refinement using a LLaMA-3 model and translation into Turkish.
🔍 Key Components:

- Model Inference: Predicts answer spans for given questions using the same TransformerQA architecture.

- Context Retrieval: Chooses appropriate context from:

*Dataset (turkish_QA_law_dataset.json), or

*Fallback files (kanun_is.txt, kanun_ceza.txt) based on question content.

- LLM Integration: Uses Groq's LLaMA 3 API to rephrase raw predictions.

- Translation: Uses deep_translator to convert the refined English output back into Turkish.

#🚀 How to Use
1. Train the model:

python model_training.py

2. Generate an answer for a question:

python post_processing_model.py

#⚙️ Requirements

Install dependencies:

    pip install torch transformers groq deep-translator

#🔐 API Key

The Groq API key is hardcoded in the script for demo purposes. For production, I recommend you to store it securely using environment variables or .env files.

You can get it free from here (https://console.groq.com/home)

#📌 Notes

This model performs best on questions found in the training dataset.

Post-processing with LLaMA3 improves clarity and coherence.

Predictions are presented in fluent Turkish through automatic translation.
