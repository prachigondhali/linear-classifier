import tensorflow as tf
import pandas as pd
from flask import Flask,request,jsonify


app = Flask(__name__)

COLUMNS=["cpu_util","dev_status"]
FEATURES=["cpu_util"]
LABEL = "dev_status"

def input_fn_train(data_set, num_epochs=100, shuffle=True,batch_size=10):
	return tf.estimator.inputs.pandas_input_fn(
			x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
			y=pd.Series(data_set[LABEL].values),
			num_epochs=num_epochs,
			shuffle=shuffle)
	
def input_fn_eval(data_set, num_epochs=1, shuffle=False,batch_size=1):
	return tf.estimator.inputs.pandas_input_fn(
			x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
			y=pd.Series(data_set[LABEL].values),
			num_epochs=num_epochs,
			shuffle=shuffle)

#def main(unused_argv):
training_set = pd.read_csv("train_Data_1.csv",header=None , names=COLUMNS)
test_set = pd.read_csv("test_Data_1.csv",header=None, names=COLUMNS)

cpuUtil=tf.feature_column.numeric_column("cpu_util")

feature_cols = [cpuUtil]
estimator = tf.estimator.LinearClassifier(feature_columns=feature_cols,optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,l1_regularization_strength=0.001))

estimator.train(input_fn=input_fn_train(training_set))
result=estimator.evaluate(input_fn=input_fn_eval(test_set))

print("Result:",result,"\n")


result=estimator.predict(input_fn=input_fn_eval(training_set[0:40]))
list_result=list(result)
for i in range( len(list_result)):
	print(i+1," ",list_result[i]["classes"] )
	
@app.route('/predict',methods=['POST'])
def prediction():
	data = request.get_json()
	#data = [{'cpu_util': 141, 'dev_status': 0  }]
	input_data=pd.DataFrame(data)
	result=estimator.predict(input_fn=input_fn_eval(input_data))
	list_result=list(result)
	for i in range( len(list_result)):
		print(i+1," ",list_result[i]["classes"] )
		return jsonify({"results": str(list_result[i]["classes"])})

if __name__ == "__main__":
	app.run(host='0.0.0.0',port=9000)

