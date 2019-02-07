from flask import Flask
from flask import render_template
from flask import request
from ChurnModel import ChurnModel as cm

import pandas as pd

app = Flask(__name__)
model = cm()


@app.route("/")
def hello():
	classifiers = model.classifiers
	return render_template('app.html', classifiers=classifiers)


@app.route("/classifier", methods=['get'])
def getClassifier():
	message = "Select Classifier Type"
	classifiers = model.classifiers
	return render_template('app.html', classifiers=classifiers)


@app.route("/classifier", methods=['post'])
def postClassifier():
	selected_classifier = request.form.get("classifier")
	res = model.train(selected_classifier)
	if len(res) == 0:
		message = "training error"
		classifiers = model.classifiers
		return render_template('app.html', classifiers=classifiers, message=message)

	score = res['score']
	predicted_score = res['predicted_score']
	confusion_matrix = pd.DataFrame(res['confusion_matrix']).to_html()
	return render_template('train_result.html',
						   selected_classifier=selected_classifier,
						   score=score,
						   confusion_matrix=confusion_matrix,
						   predicted_score=predicted_score)


@app.route("/predict", methods=['get'])
def getPredict():
	labels = model.getLabels()
	return render_template('predict.html', labels=labels)


@app.route("/predict", methods=['post'])
def postPredict():
	checks = {
		'MonthlyCharges': [float(request.form.get("MonthlyCharges"))],
		'SeniorCitizen': [float(request.form.get("SeniorCitizen"))],
		'Dependents': [request.form.get("Dependents")],
		'PhoneService': [request.form.get("PhoneService")],
		'MultipleLines': [request.form.get("MultipleLines")],
		'InternetService': [request.form.get("InternetService")],
		'OnlineSecurity': [request.form.get("OnlineSecurity")],
		'OnlineBackup': [request.form.get("OnlineBackup")],
		'TechSupport': [request.form.get("TechSupport")],
		'StreamingTV': [request.form.get("StreamingTV")],
		'StreamingMovies': [request.form.get("StreamingMovies")],
		'PaperlessBilling': [request.form.get("PaperlessBilling")]
	}

	if model.predict(checks)[0] == 1:
		message = "You clint will churn"
	else:
		message = "Your clint will not churn"

	labels = model.getLabels()
	return render_template('predict.html', labels=labels, message=message)


if __name__ == "__main__":
	app.run()
