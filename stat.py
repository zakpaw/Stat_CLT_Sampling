import streamlit as st
from streamlit_folium import folium_static

import plotly.figure_factory as ff

import folium
from folium.plugins import HeatMap

from scipy.stats import poisson
import pandas as pd
import numpy as np
np.random.seed(420)

sizes = [10, 100, 1000, 10000]
measures = {'Mean':np.mean, 'STD':np.std, 'Median':np.median}

@st.cache()
def preper_data(fun):
    global sizes
    sample_means = [[] for i in range(len(sizes))]
    for j, size in enumerate(sizes):
        means = [fun(np.random.randint(0, 7, 10)) for _ in range(size)]
        sample_means[j].append(means)
    return sample_means


data = []
for f in measures:
    data.append(preper_data(measures[f]))

'''
# CLT, Sampling & Sample Statistics
#### Paweł Żak, Marcin Tyszkiewicz
---  
---  
## 1. Sampling  
- What is sampling?  
- Random sampling, Systematic sampling
- Stratified sampling
- Cluster sampling

## 2. Central Limit Theorem (CLT)
- The Central Limit Theorem states that the sampling distribution of the  \
    sample means approaches a normal distribution as the sample size gets \
    larger — no matter what the shape of the population distribution.
'''
st.image('CLT.png', use_column_width=True)
'''
- Different from the Law of Large numbers!!!
'''
st.write('<br><br>', unsafe_allow_html=True)

st.sidebar.write('### Choose Size')
size = st.sidebar.selectbox('', list(range(len(sizes))), format_func=lambda x: sizes[x])

st.sidebar.write('### Choose measure')
measure = st.sidebar.selectbox('', list(measures.keys()))

st.write(f'## Visualization --> {measure}')

helper = list(measures.keys())
fig = ff.create_distplot(data[helper.index(measure)][size],
                         [measure],
                         bin_size=0.2,
                         show_curve=False,
                         colors=['#FA8072'])
st.plotly_chart(fig)

with st.echo():
    die = pd.Series(list(range(1, 7)))
    sample_means = []
    for i in range(0): 
        random_sample = die.sample(5, replace=True)
        sample_means.append(random_sample.mean())
        
st.write('<br><br>', unsafe_allow_html=True)
'''
## 3. Poisson Distribution
'''
with st.echo():
    sample = poisson.rvs(mu=2, size=10000, random_state=13)

fig1 = ff.create_distplot([sample],
                          [''],
                          bin_size=0.1,
                          show_curve=False,
                          colors=['#FA8072'])
st.plotly_chart(fig1)


st.write('<br><br>', unsafe_allow_html=True)
'''
## 4. Emergency - 911 Calls from montgomery country
'''
df = pd.read_csv('https://raw.githubusercontent.com/shoaibb/911-Calls-Data-Analysis/master/911.csv')
st.dataframe(data=df.head())

'''
### Calls per day
'''
calls = df[:100_000]
calls_by_date = calls.set_index(pd.DatetimeIndex(calls['timeStamp']))['timeStamp']
calls_per_day = calls_by_date.groupby(calls_by_date.index.date).count()
st.dataframe(data=calls_per_day)
'''
### Density of Accidents
'''
accdf = df.groupby(['lat', 'lng'])['lat'].count()
accdf = accdf.to_frame()
accdf.columns.values[0] = 'count1'
accdf = accdf.reset_index()
lats = accdf[['lat', 'lng', 'count1']].values.tolist()

hmap = folium.Map(location=[40.4, -75.2], zoom_start=9, )
hmap.add_child(HeatMap(lats, radius=5))
folium_static(hmap)


'''
***The Law of Large Numbers*** says that as one draws larger and larger samples \
from a population with mean μ and finite variance, the sample mean will \
converge towards μ. More formally, if Xn is a sample of size n drawn iid \
from the population, then:
'''
st.latex('lim_{n->\infty} X^n = μ')
'''
This law is useful for estimating the average value of \
a random variable in a population.  
'''
st.write(f'Calls mean:\t calls_per_day.mean() = {calls_per_day.mean()}')
with st.echo():
    @st.cache()
    def calculate_erros():
        errors = []
        for size in range(10, 1000, 10):
            sample_means = []
            for _ in range(size):
                cur_sample = calls_per_day.sample(size, replace=True)
                sample_means.append(np.mean(cur_sample))
            errors.append(abs(calls_per_day.mean()-np.mean(sample_means)))
        return errors


'''
### Errors:  
'''
st.line_chart(calculate_erros())

st.write('<br><br>', unsafe_allow_html=True)
'''
## 5. Aplications
### ...
'''
st.write('<br><br><br>', unsafe_allow_html=True)

'''
# Thank you!  
## PYTHON LIBRARY:    STREAMLIT
'''
