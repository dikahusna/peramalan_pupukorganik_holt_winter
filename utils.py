from math import sqrt

def holt_winter(data, period, alpha, beta, gamma):

	# Mendefinisikan list kosong atau nilai awal
	level = []
	trend = []
	seasonal = []
	squared_error = []
	forecast_list = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	# Menghitung nilai initial level
	initial_level = sum(data[:period])/period
	level.append(initial_level)


	# Menghitung nilai initial trend
	first_two_season = data[:period*2]
	selisih = 0
	for s in range(period):
		diff_calculate = (first_two_season[s+period]-first_two_season[s])/period
		selisih += diff_calculate
	initial_trend = selisih/period
	trend.append(initial_trend)



	for x in range(len(data)):

		if x < period:
			# Mencari initial seasonal
			seasonal.append(data[x]/initial_level)
		else:
			# Menghitung Level
			level.append(alpha*(data[x]/seasonal[x-period])+(1-alpha)*(level[x-period]+trend[x-period]))

			# Menghitung Trend
			trend.append(beta*(level[x-period+1]-level[x-period])+(1-beta)*trend[x-period])

			# Menghitung Seasonal
			seasonal.append(gamma*(data[x]/level[x-period+1])+(1-gamma)*seasonal[x-period])

			# Menghitung Forecastorecast_list.append(forecast)
			forecast = (level[x-period]+1*trend[x-period])*seasonal[x-period]
			forecast_list.append(forecast)
			# Menghitung Squared Error
			squared_error.append((forecast-data[x])**2)

			# Menghitung MSE dan RMSE
			mse = sum(squared_error)/len(squared_error)
			rmse = sqrt(mse)


	# Mengembalikan nilai trend, level, seasonal, mse, dan rmse
	return {
		"trend" : trend[-1],
		"level" : level[-1],
		"seasonal" : seasonal[-period-1:],
		"mse" : mse,
		"rmse" : rmse,
		"forecast": forecast_list
	}

def holt_winter_predict(model, period):

	# Mendefinisikan list untuk menampung hasil prediksi
	prediction = []

	# Melakukan prediksi selama 1 periode musim kedepan
	for y in range(1,period+1):
		prediction.append((model["level"]+y*model["trend"])*model["seasonal"][-period-1+y])

	# Mengembalikan list hasil prediksi
	return prediction

def holt_winter_grid_search(data, period, range_list):

	# Mendefinisikan nilai awal
	all_result = []
	best = {
		"alpha": 0,
		"beta": 0,
		"gamma": 0,
		"mse": 0,
		"rmse": 0
	}

	for g in range_list:
		for b in range_list:
			for a in range_list:
				model = holt_winter(data=data, period=period, alpha=a, beta=b, gamma=g)
				if ((g==range_list[0] and b==range_list[0] and a==range_list[0]) or (model["rmse"] < best["rmse"])):
					best = {
						"alpha": a,
						"beta": b,
						"gamma": g,
						"mse": model["mse"],
						"rmse": model["rmse"]
					}

				all_result.append([a, b, g, model["mse"], model["rmse"]])

	return {
		"all_result" : all_result,
		"best": best
	}