from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load tokenizers and config
with open("eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer = pickle.load(f)

with open("fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)

with open("config.json") as f:
    config = json.load(f)

max_fr_len = config["max_fr_len"]
english_vocab_size = len(eng_tokenizer.word_index)
french_vocab_size = len(fr_tokenizer.word_index)

# Rebuild the exact same architecture
model = Sequential()
model.add(Embedding(input_dim=english_vocab_size+1, output_dim=64, input_length=max_fr_len))
model.add(GRU(128, return_sequences=True))
model.add(Dense(french_vocab_size+1, activation='softmax'))
model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(0.001), metrics=['accuracy'])

# Build then load weights
model.build(input_shape=(None, max_fr_len))
model.load_weights("translation_model.weights.h5")

reverse_fr_index = {v: k for k, v in fr_tokenizer.word_index.items()}

def translate(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_fr_len, padding='post')
    prediction = model.predict(padded)
    words = []
    for token in np.argmax(prediction[0], axis=1):
        if token in reverse_fr_index:
            words.append(reverse_fr_index[token])
    result = " ".join(words)
    if not result.strip():
        return "(No translation found — try a sentence with common words like locations, seasons, or fruits)"
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    translation = None
    sentence = ""
    if request.method == "POST":
        sentence = request.form.get("sentence", "")
        if sentence:
            translation = translate(sentence)
    return render_template("index.html", translation=translation, sentence=sentence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)