import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import argparse
from datetime import datetime
import plotly.graph_objects as go
import os
from sklearn.metrics import confusion_matrix

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN JIND')
parser.add_argument('--file', default="datasets/human_blood_integrated_01/train.pkl", type=str,
					help='path to train data frame with labels')

def main():
	args = parser.parse_args()
	metadata = pd.read_pickle(args.file)
	print(metadata.columns)

	opacity = 0.4

	columns = ['raw_predictions', 'predictions', 'labels']

	raw = columns[0]
	pred = columns[1]
	label = "labels"

	metadata[label] = metadata[label].cat.set_categories(set(metadata[label]))
	# print((metadata[label].value_counts()))

	# truecelltypes = list(set(metadata[label]))
	truecelltypes = list(metadata[label].value_counts().index)
	# truecelltypes.sort()
	predcelltypes = truecelltypes + ["Unassigned"]

	print(list(metadata[label].value_counts().index))

	labels = list(metadata[label])
	preds = list(metadata[pred])

	cfmt = confusion_matrix(labels, preds, labels=predcelltypes)[:]
	print(cfmt)
	rows, cols = np.nonzero(cfmt)

	print(rows, cols)

	colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
				'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
				'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
				'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
				'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
	
	colors_opaque = ['rgba' + color[3:-1] + ', {})'.format(opacity) for color in colors]

	layout = go.Layout(
			autosize=False,
			width=500,
			height=500,
			font_family="Times New Roman",
			# title_text="Pancreas Barron16 - Segerstolpe16",
			title_text="PBMC Zheng17",
			title_x=0.5,
			font_size=16,

			xaxis= go.layout.XAxis(linecolor = 'black',
								linewidth = 1,
								mirror = True),

			yaxis= go.layout.YAxis(linecolor = 'black',
								linewidth = 1,
								mirror = True),

			margin=go.layout.Margin(
				l=40,
				r=40,
				b=10,
				t=40,
				pad = 4
			),
			annotations=[
						dict(
							x=-0.08,
							y=0.5,
							showarrow=False,
							text="True Cell Types",
							textangle=-90,
							xref="paper",
							yref="paper",
							font=dict(
							family="Times New Roman",
							size=20,
							# color="#ffffff"
							),
						),
						dict(
							x=1.08,
							y=0.5,
							showarrow=False,
							text="JIND Predictions with Rejection",
							textangle=-90,
							xref="paper",
							yref="paper",
							font=dict(
							family="Times New Roman",
							size=20,
							# color="#ffffff"
							),
						)
						],
			)

	fig = go.Figure(data=[go.Sankey(
								node = dict(
								pad = 15,
								thickness = 20,
								line = dict(color = "gray", width = 0.1),
								label = truecelltypes + predcelltypes,
								color = colors[:len(truecelltypes)] + colors[:len(predcelltypes)]
								),
								# arrangement="snap",
								# orientation="h",
								link = dict(
											source = rows, # indices correspond to labels, eg A1, A2, A2, B1, ...
											target = cols + len(truecelltypes),
											value = cfmt[rows, cols],
											color = np.array(colors_opaque)[rows]
											)
								)
						],
					layout = layout
					)

	# fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
	# fig.write_html('first_figure.html', auto_open=True)
	path = os.path.dirname(args.file)
	fig.write_image("{}/{}_Sankey.pdf".format(path, os.path.splitext(os.path.basename(args.file))[0]))





if __name__ == "__main__":
	main()
