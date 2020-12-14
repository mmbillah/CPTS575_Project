##########################
#LOAD REQUIRED LIBRARIES
##########################

import os
import time

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


####################
#BASIC DATA OVERVIEW
####################

#load datasets
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

#label each observation as 'train' or 'test'
#to keep track in the combined dataframe
train['dataset'] = 'Train'
test['dataset'] = 'Test'

#combine the train and test set
df = pd.concat([train, test])

# preview the train data
train.head()

# preview the test data
test.head()

#print train data rows and columns
print('Number of rows in training set:', train.shape[0])
print('Number of columns in training set:', train.shape[1] - 1)

#print test data rows and columns
print('Number of rows in test set:', test.shape[0])
print('Number of columns in test set:', test.shape[1] - 1)

#print the combined dataset info
df.info()

#################
#EDA
#################

#set initial plot dimensions
cp_width = 500
cp_height = 400
scatter_size = 600
WIDTH=800

#count for cp_type in both train and test sets
ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

#select column names
ds.columns = ['cp_type', 'Dataset', 'Count']

#define the figure
fig = px.bar(
    ds,
    x='cp_type', 
    y="Count", 
    color='Dataset',
    barmode='group',
    orientation='v', 
    title='Treatment types', 
    width=400,
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)
fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black",
    layout_showlegend=False
)
fig.update_layout(
    title={
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

#show the figure
fig.show()
#wrie images in pdf and jpeg formats
fig.write_image('cp_types.pdf',format='pdf')
fig.write_image('cp_types.jpeg',format='jpeg')

#do the same for cp_time
ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_time', 'Dataset', 'Count']

fig = px.bar(
    ds, 
    x='cp_time', 
    y="Count", 
    color='Dataset',
    barmode='group',
    orientation='v', 
    title='Treatment times', 
    width=400,
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)
fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black"
)
fig.update_layout(
    title={
        #'text': "Plot Title",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_yaxes(title='', visible=True, showticklabels=True)
fig.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="center",
    x=0.5,
    orientation="h"
))

#show figure
fig.show()
#save figure
fig.write_image('cp_times.pdf',format='pdf')
fig.write_image('cp_times.jpeg',format='jpeg')


#do the same for cp_dose
ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_dose', 'Dataset', 'Count']

fig = px.bar(
    ds, 
    x='cp_dose', 
    y="Count", 
    color='Dataset',
    barmode='group',
    orientation='v', 
    title='Treatment doses', 
    width=400,
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black"
)
fig.update_layout(
    title={
        #'text': "Plot Title",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_yaxes(title='', visible=True, showticklabels=True)
fig.update(layout_showlegend=False)

#show the figure
fig.show()
#save the figure
fig.write_image('cp_doses.pdf',format='pdf')
fig.write_image('cp_doses.jpeg',format='jpeg')

#show the three categorical variables in a sunburst chart
ds = df[df['dataset']=='Train']
ds = ds.groupby(['cp_type', 'cp_time', 'cp_dose'])['sig_id'].count().reset_index()

ds.columns = ['cp_type', 'cp_time', 'cp_dose', 'Count']

fig = px.sunburst(
    ds, 
    path=[
        'cp_type',
        'cp_time',
        'cp_dose' 
    ], 
    values='Count', 
    title='Sunburst chart for all categorical variables',
    width=600,
    height=600,
    color_discrete_sequence=px.colors.qualitative.D3
)
fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black"
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_yaxes(title='', visible=True, showticklabels=True)

#show figure
fig.show()
#save figure
fig.write_image('sunburst.pdf',format='pdf')

#correlation plot for the cell viability features
train_columns = train.columns.to_list()
#create two different lists, one for gene expression and one for cell viability
g_list = [i for i in train_columns if i.startswith('g-')]
c_list = [i for i in train_columns if i.startswith('c-')]

columns = g_list + c_list
cols = ['cp_time'] + columns
all_columns = list()
for i in range(0, len(cols)-1):
    for j in range(i+1, len(cols)):
        if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:
            all_columns = all_columns + [cols[i], cols[j]]

all_columns = list(set(all_columns))
print('Number of columns:', len(all_columns))

#plot the correlations for the top correlated features
data = df[all_columns]
del data['g-37']
del data['g-50']
fig = plt.figure(figsize=(13,8))
sns.heatmap(data.corr(), alpha=0.8)
plt.rcParams["font.family"] = "Times New Roman"
plt.title('Correlation: Cell viability', fontsize=18, color='black')
plt.xticks()
plt.yticks()
plt.show()
fig.savefig('corr.pdf',format='pdf')

# Targets analysis
train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

print('Number of rows: ', train_target.shape[0])
print('Number of cols: ', train_target.shape[1])

train_target.head()

#top 25 targets that have the most positive outcomes
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = [
    'Column name', 
    'Nonzero records'
]

x = x.tail(20)

fig = px.bar(
    x, 
    x='Nonzero records', 
    y='Column name', 
    orientation='h', 
    title='Top 25 columns', 
    width=700,
    height=600,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black"
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()
fig.write_image('top25.pdf',format='pdf')

#find bottom 20 targets that have the least positive outcomes
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = [
    'Column name', 
    'Nonzero records'
]

x = x.tail(20)

fig = px.bar(
    x, 
    x='Nonzero records', 
    y='Column name', 
    orientation='h', 
    title='Bottom 25 columns', 
    width=700,
    height=600,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black"
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()
fig.write_image('bottom25.pdf',format='pdf')

#number of activated targets for each sample
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = [
    'Count of drug samples', 
    'Count of labels'
]

data = data.groupby(['Count of labels'])['Count of drug samples'].count().reset_index()

fig = px.bar(
    data, 
    y=data['Count of drug samples'], 
    x="Count of labels", 
    title='Number of activated targets for each sample', 
    width=600, 
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black",
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0.0,
        dtick = 1.0)
)

fig.show()
fig.write_image('labelcounts.pdf',format='pdf')

#percentage of activated target counts
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = [
    'row', 
    'count'
]

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.pie(
    data, 
    values=100 * data['row'] / len(train_target), 
    names="count", 
    title='Percentage of activated target counts', 
    width=WIDTH, 
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black",
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.7,
    xanchor="right",
    x=0.9,
    orientation="v"
))

fig.show()
fig.write_image('labelpercent.pdf',format='pdf')

#percentage of drug groups in target columns
target_columns = train_target.columns.tolist()
target_columns.remove('sig_id')
last_term = dict()

for item in target_columns:
    try:
        last_term[item.split('_')[-1]] += 1
    except:
        last_term[item.split('_')[-1]] = 1

last_term = pd.DataFrame(last_term.items(), columns=['Drug groups', 'Count percentage'])
last_term = last_term.sort_values('Count percentage')
last_term = last_term[last_term['Count percentage']>1]
last_term['Count percentage'] = last_term['Count percentage'] * 100 / 206

fig = px.bar(
    last_term, 
    x='Count percentage', 
    y="Drug groups", 
    orientation='h', 
    title='Percentage of drug groups in target columns', 
    width=WIDTH,
    height=500,
    color_discrete_sequence=px.colors.qualitative.D3
)

fig.update_layout(
    font_family="georgia",
    font_color="black",
    title_font_family="georgia",
    title_font_color="Black",
    legend_title_font_color="black",
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)

fig.show()
fig.write_image('druggroups.pdf',format='pdf')
