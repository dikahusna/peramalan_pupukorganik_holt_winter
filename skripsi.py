import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from utils import *

st.write("""
# Peramalan Pupuk Subsidi di Kabupaten Bojonegoro

""")
tab1, tab2 = st.tabs(["Data", "Forecasting Data"])

with tab1:
	model = st.radio(
		"Select model",
		('Ngambon', 'Dander','Sugihwaras','Balen','Kapas','Sukosewu','Kasiman','Kedewan','Kalitidu','Temayang','Gondang',
			'Sekar','Gayam','Ngraho','Tambakrejo','Padangan','Margomulyo','Kepohbaru','Sumberrejo','Malo','Trucuk','Ngasem',
			'Bubulan','Kanor','Bojonegoro','Purwosari','Kedungadem','Baureno')
		)
	filename = 'data pupuk organik {}.csv'.format(model)
	df = pd.read_csv(filename, delimiter=',')
	df['Tahun'] = df['Tahun'].astype(str)
	df['Bulan Tahun'] = df['Bulan'].str.cat(df['Tahun'], sep=' ')
	df = df.drop('Bulan', axis=1)
	df = df.drop('Tahun', axis=1)

	gb = GridOptionsBuilder.from_dataframe(df)
	gb.configure_pagination(
		paginationAutoPageSize=False, 
		paginationPageSize=10
	)
	gridOptions = gb.build()
	AgGrid(df, gridOptions=gridOptions)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(df["Bulan Tahun"], df["Data"])
	ax.set_title("Grafik Data")
	ax.set_xlabel("Bulan Tahun")
	ax.tick_params(labelrotation=90)
	ax.set_ylabel("Data")
	st.pyplot(fig)
with tab2:
	model = st.selectbox(
		"Pilih Kecamatan",
		('Ngambon', 'Dander','Sugihwaras','Balen','Kapas','Sukosewu','Kasiman','Kedewan','Kalitidu','Temayang','Gondang',
			'Sekar','Gayam','Ngraho','Tambakrejo','Padangan','Margomulyo','Kepohbaru','Sumberrejo','Malo','Trucuk','Ngasem',
			'Bubulan','Kanor','Bojonegoro','Purwosari','Kedungadem','Baureno'))

	# Baca data
	filename = 'data pupuk organik {}.csv'.format(model)
	df = pd.read_csv(filename, delimiter=',')

	df['Tahun'] = df['Tahun'].astype(str)
	df['Bulan Tahun'] = df['Bulan'].str.cat(df['Tahun'], sep=' ')
	df = df.drop('Bulan', axis=1)
	df = df.drop('Tahun', axis=1)

	# cari alpha beta gamma terbaik

	best_params = holt_winter_grid_search(data=df["Data"], period=12, range_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

	best_model = holt_winter(data=df["Data"], period=12, alpha=best_params["best"]["alpha"], beta=best_params["best"]["beta"], gamma=best_params["best"]["gamma"])
	prediction_result = holt_winter_predict(model=best_model, period=12)

	data = {
	'Bulan Tahun': ['Januari 2023', 'Februari 2023', 'Maret 2023', 'April 2023', 'Mei 2023','Juni 2023', 'Juli 2023', 
	'Agustus 2023', 'September 2023', 'Oktober 2023', 'November 2023', 'Desember 2023'],
	'Hasil Ramalan': prediction_result,
	}

	data_forecast = best_model["forecast"]
	df['Hasil Ramalan'] = data_forecast
	res_df = pd.DataFrame(data)
	gb = GridOptionsBuilder.from_dataframe(res_df)
	gb.configure_pagination(
		paginationAutoPageSize=False, 
		paginationPageSize=10
	)
	gridOptions = gb.build()
	AgGrid(res_df, gridOptions=gridOptions)

	st.write(f"Peramalan ini dikalkulasi menggunakan parameter **alpha = {best_params['best']['alpha']}** , **beta = {best_params['best']['beta']}** dan **gamma = {best_params['best']['gamma']}** karena menghasilkan error terendah, yaitu dengan nilai **RMSE** sebesar **{round(best_params['best']['rmse'], 2)}** dan **MSE** sebesar **{round(best_params['best']['mse'], 2)}**")

	data['Data'] = [0,0,0,0,0,0,0,0,0,0,0,0]
	res_df = pd.DataFrame(data)

	df = pd.concat([df, res_df], ignore_index=True)

	fig = plt.figure(figsize=(12, 4))
	plt.plot(df["Bulan Tahun"][:-12],df['Data'][:-12])
	plt.plot(df["Bulan Tahun"][12:],df['Hasil Ramalan'][12:])
	plt.title(f'Data Aktual dan Peramalan Penjualan Pupuk Organik di {model}')
	plt.ylabel('Jumlah')
	plt.xlabel('Bulan Tahun')
	plt.xticks(rotation=90)
	plt.legend(['Aktual', 'Peramalan'], loc='upper left')
	plt.grid(False)
	st.pyplot(fig)
	param_df = pd.DataFrame(data=best_params['all_result'],columns=['Alpha', 'Beta', 'Gamma', 'RMSE', 'MSE'])
	
	gb = GridOptionsBuilder.from_dataframe(param_df)
	gridOptions = gb.build()
	AgGrid(param_df, gridOptions=gridOptions)

	

