# __GMV의 한계__

## __특정 자산 편중__

+ 투자 비중이 배분되지 않고 일부 자산에 과도한 비중으로 투자(기대수익률, 위험측정 오차가 최적화 단계에서 증폭되기 때문)
    + 특정 자산에 편중되는지 확인 하기 위해 2022년 7월 기준 시가총액 상위 15개의 기업을 활용하여 최적화 진행
+ 과거자료만 가지고 기대수익률과 변동성을 다루고 있기 때문에 투자대상에 대한 미래가치를 장담하지 못함.
    + 2010-06-29부터 2015-06-30까지의 자료로 가중치를 구하고 2015-07-01부터 2022-06-30까지 계산한 수익률, 표준편차와 2015-07-01부터 2022-06-30까지 최적화로 구한 수익률과 표준편차를 비교


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
try:
    import pandas_datareader as web
    import cvxopt as opt
    from cvxopt import solvers

except:
    !pip install pandas_datareader 
    !pip install cvxopt
    import pandas_datareader as web

```

```python
#GMV Portfolio
ticker = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'BRK-A', 
            'TSM', 'UNH', 'JNJ', 'WMT', 'NVDA', 'PG', 'JPM', 'XOM', 'MA']
start_date = '2010-06-29'
end_date = '2015-06-30'

start_date2 = '2015-07-01'
end_date2 = '2022-06-30'

data = web.get_data_yahoo(ticker, start_date, end_date)['Adj Close'].dropna()
data2 = web.get_data_yahoo(ticker, start_date2, end_date2)['Adj Close'].dropna()

ret = data.pct_change().dropna()
ret2 = data2.pct_change().dropna()
data = data/data.iloc[0] * 100

```

```python

try :
    from scipy import optimize as op
except :
    ! pip install scipy
    from scipy import optimize as op

#GMV Portfolio
def gmv(x):
    return (np.dot(x.T, np.dot(ret.cov()* 252, x)))

def gmvportfolio(x):
    x0 = [1/len(x.columns) for i in range(len(x.columns))]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, )  #가중치의 합= 1
    bound = (0.0, 1.0)
    bounds =tuple(bound for i in range(len(x.columns))) #Long only
    options = {'ftol' : 1e-20, }

    sol = op.minimize(gmv, x0, constraints = constraints, 
                    bounds = bounds, options = options, method='SLSQP',)
    return sol

```

```python
gmv_wgt =gmvportfolio(ret).x.transpose()
gmv_ret = np.sum(ret.mean() * gmv_wgt * 252)
gmv_std = np.sqrt(gmvportfolio(ret).fun)


gmv_wgt2 = pd.DataFrame(gmv_wgt).T
gmv_wgt2.columns = ret.columns
dt2 = {"Returns" : gmv_ret, 'Stds' : gmv_std}
dt2 = pd.DataFrame(dt2, index = [0])

eff2 = pd.concat([dt2, gmv_wgt2], axis =1)

eff2.T
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>0.000</td>
    </tr>
    <tr>
      <th>GOOGL</th>
      <td>0.006</td>
    </tr>
        <tr>
      <th>MSFT</th>
      <td>0.000</td>
    </tr> 
      <th>AMZN</th>
      <td>0.030</td>
    </tr>
    <tr>
      <th>TSLA</th>
      <td>0.000</td>
    </tr>
        <tr>
      <th>BRK-A</th>
      <td>0.141</td>
    </tr> 
      <th>TSM</th>
      <td>0.033</td>
    </tr>
    <tr>
      <th>UNHL</th>
      <td>0.000</td>
    </tr>
        <tr>
      <th>JNJ</th>
      <td>0.275</td>
    </tr> 
      <th>WMT</th>
      <td>0.224</td>
    </tr>
    <tr>
      <th>NVDA</th>
      <td>0.000</td>
    </tr>
        <tr>
      <th>PG</th>
      <td>0.249</td>
    </tr> 
      <th>JPM</th>
      <td>0.000</td>
    </tr>
    <tr>
      <th>XOM</th>
      <td>0.042</td>
    </tr>
        <tr>
      <th>MA</th>
      <td>0.000</td>
    </tr> 


  </tbody>
</table>



```python
ret_test =  np.array(data2.pct_change().dropna())
wgt_test = np.array(gmv_wgt2)
var_test = np.array(data2.pct_change().dropna().cov())

ret_tp = np.sum(ret_test.mean() * wgt_test * 252)
cov_tp = np.sqrt(np.dot(wgt_test, (np.dot((var_test)*252, wgt_test.transpose()))))    
```


```python
dt3 = {"Returns" : ret_tp, 'Stds' : cov_tp[0]}
dt3 = pd.DataFrame(dt3, index = ['과거기반'])
```


```python
gmv_wgt =gmvportfolio(ret2).x.transpose()
gmv_ret = np.sum(ret2.mean() * gmv_wgt * 252)
gmv_std = np.sqrt(gmvportfolio(ret).fun)


gmv_wgt2 = pd.DataFrame(gmv_wgt).T
gmv_wgt2.columns = ret.columns
dt2 = {"Returns" : gmv_ret, 'Stds' : gmv_std}
dt2 = pd.DataFrame(dt2, index = ['최적화기반'])

eff2 = pd.concat([dt2, gmv_wgt2], axis =1)

```


```python
 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Returns</th>
      <th>Stds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>과거기반</th>
      <td>0.247</td>
      <td>0.161</td>
    </tr>
    <tr>
      <th>최적화기반</th>
      <td>0.140</td>
      <td>0.112</td>
    </tr>

  </tbody>
</table>
