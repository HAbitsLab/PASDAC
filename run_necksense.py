from Plot.plotConfusionMatrix import plotConfusionMatrix
from Plot.plotROC import plotROC
from Tools.class_settings import SETTING
from pipeline import *
from dataloader import Bulling_dataloader
import pandas as pd
import plotly.graph_objects as go
import numpy as np

setting = SETTING('DataNecksense', 'Output', '/feature')
setting.set_DATASET('chew')
setting.set_SAMPLINGRATE(20)
setting.set_SUBJECT_LIST(["P101", "P102"])
setting.VERBOSE_LEVEL = 2

# setting = SETTING('Data2R', 'Output', '/feature')
# setting.set_DATASET('gesture')
# setting.set_SAMPLINGRATE(32)
# setting.set_SUBJECT_LIST([1, 2])

dataset = Bulling_dataloader(setting)

raw_data = dataset.raw["P101"]["data"]
print(raw_data)

raw_data = dataset.raw["P101"]["data"][1]
print(raw_data)

# Viewing data for a particular participant
def show_data(df,title="",labels=None,color="LightSalmon"):

  # Create figure
  fig = go.Figure()

  for c in ['proximity']:
    fig.add_trace(
        go.Scatter(x=df.index.values, y=df[c].values,name=c)
        )

  # Set title
  fig.update_layout(
      title_text=title,
      width=900,
      height=400
  )

  # Add range slider
  fig.update_layout(
      xaxis=go.layout.XAxis(
          rangeslider=dict(
              visible=True
          ),
          type="linear"
      )
  )



  # plotting the labels
  if labels is not None:
    labels_shape = []
    for i in labels.index:
      labels_shape += [go.layout.Shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=str(labels.loc[i].Start),
            y0=0,
            x1=str(labels.loc[i].End),
            y1=1,
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=2,
        )]

    fig.update_layout(shapes=labels_shape)

  fig.show()

show_data(raw_data,"Raw Data")