from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load your pre-trained model
with open('spam_detect_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load CountVectorizer (for transforming input text)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']

        # Preprocess the input message
        preprocessed_message = preprocess_text(message)

        # Transform the preprocessed message using the vectorizer
        message_vectorized = vectorizer.transform([preprocessed_message])

        # Make a prediction
        prediction = model.predict(message_vectorized)

        # Interpret the prediction
        result = 'ham' if prediction[0] == 0 else 'spam'

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
