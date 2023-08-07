'''
This module analyzes collected data in order to ID early patterns and extract statistical insights.
'''

import pandas as pd # for working with CSVs
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
import numpy as np # for data manipulation
import os # for file pathing



class SensorEDA:
	'''
	Exploratory data analysis for sensor data.
	'''
	def __init__(self, file):
		self.data = self._load_data(file)

	def _load_data(self, file):
		'''
		Load data from a CSV file into a pandas dataframe.
		'''
		return pd.read_csv(file)

	def stats(self):
		'''
		Display basic statistics about the data.
		'''
		print(self.data.describe()) # basic statistics
		print("\nNaNs:")
		print(self.data.isna().sum()) # number of NaNs
		print("\nNulls:")
		print(self.data.isnull().sum()) # number of nulls

	def visualize(self):
		'''
		Visualize the data.
		'''
		plt.figure(figsize=(16, 8)) # set figure size (rect)

		for col in self.data.columns: # for each column in all the columns in our dataset ...
			if col not in ["timestamp", "label"]: # ... if the column is not the timestamp or label ...
				sns.lineplot(x="timestamp", y=col, label=col) # ... plot the column

		plt.title("Sensor data/time")
		plt.xlabel("Time")
		plt.ylabel("Sensor data")
		plt.legend() # this toggles the labels in the plot
		plt.show()

	def detect_anomalies(self):
		'''
		Detect anomalies in the data.
		'''
		Q1 = self.data.quantile(0.25) # 25th percentile - median of the lower half of the data
		Q3 = self.data.quantile(0.75) # 75th percentile - median of the upper half of the data
		IQR = Q3 - Q1 # interquartile range - range between the 25th and 75th percentile (variability of the data)

		lowerBound = Q1 - 1.5 * IQR # lower bound - anything below this is an outlier - the 1.5 is a standard threshold
		upperBound = Q3 + 1.5 * IQR # upper bound - anything above this is an outlier - the 1.5 is a standard threshold

		outliers = (self.data < lowerBound) | (self.data > upperBound) # outliers are anything below the lower bound OR above the upper bound
		print("\nOutliers:")
		print(outliers.sum()) # number of outliers

	def eda(self):
		'''
		Perform exploratory data analysis.
		'''
		self.stats()
		self.visualize()
		self.detect_anomalies()