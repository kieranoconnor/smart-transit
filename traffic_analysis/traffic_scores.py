import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TRAFFIC_FILES = ["data_1.csv", "data_2.csv"]
KWH_FILE = "block_weather_avg_test.csv"

def select_k_best_segments(k=10):
	from sklearn.feature_selection import SelectKBest
	traffic = load_all_traffic()
	kwh = load_kwh()
	meta = pd.read_csv('metadata.csv')

	seg_day_scores = traffic.groupby(["Date", "Segment ID"]).agg({"Score": 'mean'})
	seg_day_scores = seg_day_scores.reset_index()
	seg_day_scores = seg_day_scores.pivot(index="Date", columns="Segment ID", values=["Score"])

	kwh = kwh[['Date', 'kwh']]
	merged = merged.fillna(merged.mean())

	merged = pd.merge(kwh, seg_day_scores, on="Date")
	X = merged.drop(['kwh', 'Date'], axis=1)
	y = merged['kwh']
	Xn = X.drop(X[X.columns[X.isna().any(axis=0)]].columns, axis=1) # drops cols with all na
	
	model = SelectKBest(k=k)
	model.fit(Xn, y)
	X_new = Xn.iloc[:, model.get_support()]

	seg_imp = pd.DataFrame(data={'Segment ID': X_new.columns.get_level_values(1)})
	matched = pd.merge(seg_imp, meta, on="Segment ID")
	return matched


def get_k_best_segments2():
	meta = pd.read_csv('metadata.csv')
	segments = pd.DataFrame(data={'Segment ID': [ 429474946, 441557292, 449837826, 449839354, 464456709,
		1626648917, 1626679766, 1626700859, 1626700879, 1626774427]})
	seg_meta = pd.merge(segments, meta, on='Segment ID')
	return seg_meta

def get_relevant_segments():
	# load the files
	meta = pd.read_csv("metadata.csv")
	gps = pd.read_excel("77FEB2020.XLSX")

	# split cross streets into road names
	roads = gps["MAIN_CROSS_STREET"].str.split(" + ", regex=False)
	road_arr = []

	for road in roads:
		road_arr.extend(road)
	# get all unique sesgments
	unique_roads = pd.Series(road_arr).unique()

	# use regex to find metadata intersections that have segment names
	matches_regex = "|".join(unique_roads)

	matches_bools = meta["Intersection"].str.contains(matches_regex, na=False, regex=True)
	matched_rows = meta[matches_bools]
	return matched_rows

def get_all_segments():
	return pd.read_csv("metadata.csv")

def load_all_traffic():
	data = None

	relevant_segments = get_all_segments()
	for file in TRAFFIC_FILES:
		if data is None:
			data = pd.read_csv(file)
			data = data[['Date Time', 'Segment ID', 'Speed(miles/hour)', 'Ref Speed(miles/hour)', 'CValue']]
			data = data.merge(relevant_segments, on="Segment ID")
			data["Score"] = data["Speed(miles/hour)"] / data["Ref Speed(miles/hour)"]
			data["Date Time"] = pd.to_datetime(data["Date Time"]).dt.tz_localize(None)
			data["Date"] = data["Date Time"].dt.date
			data["Time"] = data["Date Time"].dt.time
		else:
			add = pd.read_csv(file)
			add = add[['Date Time', 'Segment ID', 'Speed(miles/hour)', 'Ref Speed(miles/hour)', 'CValue']]
			add = add.merge(relevant_segments, on="Segment ID")
			add["Score"] = add["Speed(miles/hour)"] / add["Ref Speed(miles/hour)"]
			add["Date Time"] = pd.to_datetime(add["Date Time"]).dt.tz_localize(None)
			add["Date"] = add["Date Time"].dt.date
			add["Time"] = add["Date Time"].dt.time
			data = pd.concat([data, add], axis=0)
	return data

def load_relevant_traffic(kind='all', k=10):
	data = None
	kinds = ['all', 'regex', 'selectk']
	if kind not in kinds:
		raise ValueError('Invalid kind. Expected one of: %s' % kinds)
	relevant_segments = get_relevant_segments() if kind == 'regex' else get_k_best_segments2() if kind == 'selectk' else get_all_segments()
	for file in TRAFFIC_FILES:
		if data is None:
			data = pd.read_csv(file)
			data = data[['Date Time', 'Segment ID', 'Speed(miles/hour)', 'Ref Speed(miles/hour)', 'CValue']]
			data = data.merge(relevant_segments, on="Segment ID")
			data["Score"] = data["Speed(miles/hour)"] / data["Ref Speed(miles/hour)"]
			data["Date Time"] = pd.to_datetime(data["Date Time"]).dt.tz_localize(None)
			data["Date"] = data["Date Time"].dt.date
			data["Time"] = data["Date Time"].dt.time
		else:
			add = pd.read_csv(file)
			add = add[['Date Time', 'Segment ID', 'Speed(miles/hour)', 'Ref Speed(miles/hour)', 'CValue']]
			add = add.merge(relevant_segments, on="Segment ID")
			add["Score"] = add["Speed(miles/hour)"] / add["Ref Speed(miles/hour)"]
			add["Date Time"] = pd.to_datetime(add["Date Time"]).dt.tz_localize(None)
			add["Date"] = add["Date Time"].dt.date
			add["Time"] = add["Date Time"].dt.time
			data = pd.concat([data, add], axis=0)
	return data

def load_kwh():
	kwh = pd.read_csv(KWH_FILE, index_col=0)
	kwh["Date Time"] = pd.to_datetime(kwh["Date"]).dt.tz_localize(None)
	kwh["Date"] = kwh["Date Time"].dt.date

	return kwh

def kwh_vs_score(traffic, kwh):
	kwh_by_date = kwh.groupby("Date").agg({"kwh": "mean"})

	confident_traffic = traffic#.loc[traffic["CValue"] > 0.9]
	mean_scores = confident_traffic.groupby(["Date", "Segment ID"])["Score"].apply(np.mean).reset_index()
	day_scores = mean_scores.groupby("Date").agg({"Score": "mean"})

	merged = pd.merge(kwh_by_date, day_scores, on="Date")

	sns.scatterplot(merged, x="Score", y="kwh")
	
	return merged

def linreg(traffic_scores):
	from scipy.stats import linregress
	x = traffic_scores["Score"]
	y = traffic_scores['kwh']
	gradient, intercept, r_value, p_value, std_err = linregress(x, y)
	mn=np.min(x)
	mx=np.max(x)
	x1=np.linspace(mn,mx,500)
	y1=gradient*x1+intercept
	plt.plot(x,y,'ob')
	plt.plot(x1,y1,'-r')
	plt.ylabel("KWH")
	plt.xlabel("Score")
	plt.show()

	return r_value