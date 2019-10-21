from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def convert_raw_data_to_timeseries(data, n_in=1, n_out= 1, dropnan=True):
	"""
	:param data: DataFrame
	:param n_in:
	:param n_out:
	:param dropnan:
	:return:
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('variable%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('variable%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('variable%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#LSTM
def runModel(scaledArray, n_features, n_days_to_feed_model, n_days_to_predict):
	"""
	:param scaledArray: numpy.Array
	:param n_features: Int
	:param n_days_to_feed_model: Int
	:param n_days_to_predict: Int
	:return: model, keras.callbacks.callbacks.History
	"""
	reframed = convert_raw_data_to_timeseries(scaledArray, n_days_to_feed_model, n_days_to_predict)

	reframedValues = reframed.values
	n_train_days = 365 * 3
	train = reframedValues[:n_train_days, :]
	test = reframedValues[n_train_days:, :]

	n_obs = n_days_to_feed_model * n_features
	train_X, train_y = train[:, :n_obs], train[:, -n_obs]
	test_X, test_y = test[:, :n_obs], test[:, -n_obs]

	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], n_days_to_feed_model, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_days_to_feed_model, n_features))

	# network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')

	# fit network
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	return model, history, test_X, test_y

def preprocessModelDf(modelDf, categorical_feature_index_list = []):
	"""
	:param modelDf: DataFrame
	:param categorical_feature_index_list: List
	:return: scaled DataFrame
	"""

	values = modelDf.values
	# integer encode direction

	if len(categorical_feature_index_list):
		encoder = LabelEncoder()
		for categorical_feature in categorical_feature_index_list:
			values[:, categorical_feature] = encoder.fit_transform(values[:, categorical_feature])

	# ensure all data is float
	values = values.astype('float32')
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	return scaled


def plotLossHistory(history):

	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()


def makePrediction(test_X,test_y,model,n_days_to_feed_model,n_features):
	"""
	:param test_X: Numpy Array
	:return: Double
	"""
	scaler = MinMaxScaler(feature_range=(0, 1))
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], n_days_to_feed_model * n_features))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)
	return rmse