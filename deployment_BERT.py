import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification,DistilBertTokenizer
from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS


app = Flask(__name__)
CORS(app)

df = pd.read_csv('output.csv')

Y= list(df['label'])
le = LabelEncoder()
Y = le.fit_transform(Y)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_path = 'saved_model'
tokenizer_path = 'saved_model'

loaded_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
	try:
		if request.method == 'POST':
			print(request.get_json())
			a = request.get_json()
			name = a['text']
			new_text = [name]
			new_encodings = loaded_tokenizer(new_text, truncation=True, padding=True, return_tensors='tf')

			# Make predictions
			predictions = loaded_model.predict(new_encodings)
			predicted_labels = np.argmax(predictions.logits, axis=1)

			# Convert label indices to original labels (if you used LabelEncoder)
			predicted_labels_text = le.inverse_transform(predicted_labels)  # Assuming 'le' is still available
			print("Predicted labels (text):", predicted_labels_text)
		return({"Prediction":predicted_labels_text})
	except:
		print("error")
		return({"data":"error"})

if __name__ == "__main__":
    app.run(host='192.168.1.4', port='6001')

