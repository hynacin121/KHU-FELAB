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

    Collecting pandas_datareader
      Using cached pandas_datareader-0.10.0-py3-none-any.whl (109 kB)
    Requirement already satisfied: requests>=2.19.0 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas_datareader) (2.27.1)
    Requirement already satisfied: pandas>=0.23 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas_datareader) (1.4.2)
    Requirement already satisfied: lxml in ./opt/anaconda3/lib/python3.9/site-packages (from pandas_datareader) (4.8.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.23->pandas_datareader) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.23->pandas_datareader) (2021.3)
    Requirement already satisfied: numpy>=1.18.5 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.23->pandas_datareader) (1.21.5)
    Requirement already satisfied: six>=1.5 in ./opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas>=0.23->pandas_datareader) (1.16.0)
    Requirement already satisfied: certifi>=2017.4.17 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.19.0->pandas_datareader) (2021.10.8)
    Requirement already satisfied: idna<4,>=2.5 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.19.0->pandas_datareader) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.19.0->pandas_datareader) (1.26.9)
    Requirement already satisfied: charset-normalizer~=2.0.0 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.19.0->pandas_datareader) (2.0.4)
    Installing collected packages: pandas-datareader
    Successfully installed pandas-datareader-0.10.0



```python
#GMV Portfolio

ticker = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date = '2010-06-29'
end_date = '2022-06-30'

data = web.get_data_yahoo(ticker, start_date, end_date)['Adj Close'].dropna()

ret = data.pct_change().dropna()
data = data/data.iloc[0] * 100

```


```python
rf = web.get_data_yahoo('TLT', '2010-06-28', '2022-06-30')['Adj Close'].pct_change().dropna()
rf_rate = rf.mean() * 252
```


```python
rf_rate
```


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-06-29</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>2010-06-30</th>
      <td>98.188677</td>
      <td>97.950509</td>
      <td>98.713004</td>
      <td>100.598468</td>
      <td>99.748847</td>
    </tr>
    <tr>
      <th>2010-07-01</th>
      <td>96.998081</td>
      <td>96.748557</td>
      <td>99.356476</td>
      <td>102.163702</td>
      <td>91.921312</td>
    </tr>
    <tr>
      <th>2010-07-02</th>
      <td>96.396903</td>
      <td>96.101353</td>
      <td>99.828383</td>
      <td>100.487980</td>
      <td>80.368355</td>
    </tr>
    <tr>
      <th>2010-07-06</th>
      <td>97.056630</td>
      <td>95.995682</td>
      <td>102.187887</td>
      <td>101.335047</td>
      <td>67.434072</td>
    </tr>
    <tr>
      <th>2010-07-07</th>
      <td>100.975881</td>
      <td>99.106234</td>
      <td>104.247080</td>
      <td>104.437900</td>
      <td>66.136462</td>
    </tr>
    <tr>
      <th>2010-07-08</th>
      <td>100.749456</td>
      <td>100.506319</td>
      <td>104.718988</td>
      <td>107.006718</td>
      <td>73.084977</td>
    </tr>
    <tr>
      <th>2010-07-09</th>
      <td>101.346732</td>
      <td>102.912425</td>
      <td>104.118376</td>
      <td>107.964274</td>
      <td>72.833824</td>
    </tr>
    <tr>
      <th>2010-07-12</th>
      <td>100.437166</td>
      <td>104.748380</td>
      <td>106.520792</td>
      <td>110.035910</td>
      <td>71.368777</td>
    </tr>
    <tr>
      <th>2010-07-13</th>
      <td>98.294121</td>
      <td>107.691630</td>
      <td>107.807767</td>
      <td>113.847709</td>
      <td>75.931354</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Feasible Set
n = len(data.columns)
returns = []
stds = []
wgt = []
count = 1000000

tr = 0
```


```python
for i in range(count):
    weights = np.random.random(n)
    weights /= sum(weights)
    wgt.append(weights)
    mean = np.sum(weights * ret.mean()*252)  # *252로 1년 수익률 계산
    var = np.dot(weights.T, np.dot(ret.cov()*252, weights))
    cov = np.sqrt(var)
    returns.append(mean)
    stds.append(cov)
    tr += 1
```


```python
wgt2 = pd.DataFrame(wgt, columns = data.columns)
sharpe = np.array(returns)/np.array(stds)
tangent = (np.array(returns)- rf_rate)/np.array(stds)
dt = {'Returns':returns, 'Stds': stds, 'Sharpe' : sharpe, 'Tangent' : tangent}
dt = pd.DataFrame(dt)

eff = pd.concat([dt,wgt2], axis = 1)

eff
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Returns</th>
      <th>Stds</th>
      <th>Sharpe</th>
      <th>Tangent</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.328813</td>
      <td>0.257072</td>
      <td>1.279072</td>
      <td>1.091394</td>
      <td>0.248901</td>
      <td>0.126237</td>
      <td>0.380964</td>
      <td>0.020289</td>
      <td>0.223608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.340082</td>
      <td>0.261095</td>
      <td>1.302520</td>
      <td>1.117733</td>
      <td>0.293049</td>
      <td>0.002514</td>
      <td>0.285628</td>
      <td>0.201202</td>
      <td>0.217607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.319227</td>
      <td>0.248367</td>
      <td>1.285305</td>
      <td>1.091048</td>
      <td>0.298237</td>
      <td>0.167381</td>
      <td>0.166466</td>
      <td>0.199677</td>
      <td>0.168238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.272077</td>
      <td>0.230325</td>
      <td>1.181276</td>
      <td>0.971803</td>
      <td>0.158171</td>
      <td>0.245155</td>
      <td>0.335766</td>
      <td>0.225354</td>
      <td>0.035554</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.289280</td>
      <td>0.235519</td>
      <td>1.228263</td>
      <td>1.023410</td>
      <td>0.138590</td>
      <td>0.128189</td>
      <td>0.420677</td>
      <td>0.234355</td>
      <td>0.078189</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>0.322758</td>
      <td>0.249728</td>
      <td>1.292438</td>
      <td>1.099240</td>
      <td>0.209654</td>
      <td>0.112694</td>
      <td>0.301949</td>
      <td>0.194144</td>
      <td>0.181560</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>0.349222</td>
      <td>0.271425</td>
      <td>1.286625</td>
      <td>1.108871</td>
      <td>0.199134</td>
      <td>0.021144</td>
      <td>0.413922</td>
      <td>0.095539</td>
      <td>0.270261</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>0.389582</td>
      <td>0.319615</td>
      <td>1.218911</td>
      <td>1.067958</td>
      <td>0.160572</td>
      <td>0.277915</td>
      <td>0.002579</td>
      <td>0.141322</td>
      <td>0.417611</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>0.352367</td>
      <td>0.274801</td>
      <td>1.282261</td>
      <td>1.106691</td>
      <td>0.139329</td>
      <td>0.146620</td>
      <td>0.240847</td>
      <td>0.189859</td>
      <td>0.283344</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>0.345859</td>
      <td>0.278388</td>
      <td>1.242363</td>
      <td>1.069055</td>
      <td>0.027270</td>
      <td>0.315447</td>
      <td>0.138150</td>
      <td>0.238066</td>
      <td>0.281067</td>
    </tr>
  </tbody>
</table>
<p>1000000 rows × 9 columns</p>
</div>




```python
ret.mean()*252
ret.std()*np.sqrt(252)
```




    Symbols
    AAPL     0.282603
    GOOGL    0.263697
    MSFT     0.256018
    AMZN     0.322242
    TSLA     0.569688
    dtype: float64




```python
spot = pd.concat([ret.mean()*252, ret.std()*np.sqrt(252)], axis =1)
spot.columns = ['mean', 'std']

```


```python
def efficient_return(self, target):
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix)

        constraints = ({'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(num_assets))
        result = optimize.minimize(self.portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, 
                                   constraints=constraints)
        return result


def efficient_frontier(self, returns_range):
        efficients = []
        for ret in returns_range:
            efficients.append(self.efficient_return(ret))
        return efficients

```


```python
#Efficient Frontier

mvport = eff.iloc[eff['Stds'].idxmin()]
mvport2 = eff.iloc[eff['Sharpe'].idxmax()]
mvport3 = eff.iloc[eff['Tangent'].idxmax()]

plt.subplots(figsize= [15,15])
plt.scatter(spot['std'], spot['mean'], marker = 'o', s=100, color = 'k')
plt.scatter(eff['Stds'], eff['Returns'], marker= 'o', s = 1, alpha =0.7, c = eff['Sharpe'], cmap ='viridis')
plt.scatter(mvport[1], mvport[0], color = 'r', marker = '*', s = 150, label ="GMV")
plt.scatter(mvport2[1], mvport2[0], marker = '*', s = 150, label ="Max Sharpe Ratio")
plt.scatter(mvport3[1], mvport3[0], color = 'k', marker = '*', s = 150, label = "Tangent Portfolio")
plt.scatter(0, rf_rate, color = 'k',marker = 's', s = 150)
plt.plot([0, 2* mvport3[1],], [rf_rate, 2*mvport3[0]-rf_rate], color = 'red', label = 'CAL')
for i in range(spot.shape[0]):
    plt.text(x=spot['std'][i]+0.005, y=spot['mean'][i], s = spot.index[i], fontsize =16)
plt.colorbar()
plt.xlim(0.15,0.6)
plt.ylim(0.15,0.6)
plt.xlabel('Risk')
plt.ylabel("Return")
plt.legend(loc=2, fontsize = 20)
plt.show
```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_11_1.png)
    



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
    bound = (0.0001, 1.0)
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Returns</th>
      <td>0.25460</td>
    </tr>
    <tr>
      <th>Stds</th>
      <td>0.22598</td>
    </tr>
    <tr>
      <th>AAPL</th>
      <td>0.24666</td>
    </tr>
    <tr>
      <th>GOOGL</th>
      <td>0.30547</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.36320</td>
    </tr>
    <tr>
      <th>AMZN</th>
      <td>0.08457</td>
    </tr>
    <tr>
      <th>TSLA</th>
      <td>0.00010</td>
    </tr>
  </tbody>
</table>
</div>




```python
mvport = pd.DataFrame(mvport)

compare = pd.concat([eff2.T, mvport], axis = 1)
compare.columns = ['Opt', 'Trial']

compare
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Opt</th>
      <th>Trial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Returns</th>
      <td>0.25460</td>
      <td>0.25523</td>
    </tr>
    <tr>
      <th>Stds</th>
      <td>0.22598</td>
      <td>0.22601</td>
    </tr>
    <tr>
      <th>AAPL</th>
      <td>0.24666</td>
      <td>0.24161</td>
    </tr>
    <tr>
      <th>GOOGL</th>
      <td>0.30547</td>
      <td>0.29364</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.36320</td>
      <td>0.37389</td>
    </tr>
    <tr>
      <th>AMZN</th>
      <td>0.08457</td>
      <td>0.09033</td>
    </tr>
    <tr>
      <th>TSLA</th>
      <td>0.00010</td>
      <td>0.00051</td>
    </tr>
  </tbody>
</table>
</div>



Chapter4 CAPM


```python
#CML : 시장포트폴리오를 무위험 자산의 투자배분을 통해 나온 자본 배분선

#시장포트폴리오 : SP500

mkt_data = web.get_data_yahoo('^sp1500', '2010-06-30', end_date)['Adj Close']
mkt = mkt_data.pct_change().dropna()
mkt_data = mkt_data/mkt_data[0] * 100
mkt_return = mkt.mean() * 252
mkt_var = mkt.var() * 252
mkt_std = mkt_var ** (1/2)
```


```python
plt.subplots(figsize= [15,15])
plt.scatter(spot['std'], spot['mean'], marker = 'o', s=100, color = 'k')
plt.scatter(eff['Stds'], eff['Returns'], marker= 'o', s = 1, alpha =0.7, c = eff['Sharpe'], cmap ='viridis')
plt.scatter(mvport[1], mvport[0], color = 'r', marker = '*', s = 150, label ="GMV")
plt.scatter(mvport2[1], mvport2[0], marker = '*', s = 150, label ="Max Sharpe Ratio")
plt.scatter(mvport3[1], mvport3[0], color = 'k', marker = '*', s = 150, label = "Tangent Portfolio")
plt.scatter(0, rf_rate, color = 'k',marker = 's', s = 150)
plt.scatter(mkt_std, mkt_return, color = 'g', s= 100, label ='Market Portfolio')
plt.plot([0, 2* mvport3[1],], [rf_rate, 2*mvport3[0]-rf_rate], color = 'red', label = 'CAL')
for i in range(spot.shape[0]):
    plt.text(x=spot['std'][i]+0.005, y=spot['mean'][i], s = spot.index[i], fontsize =16)

plt.plot([0, 5*mkt_std], [rf_rate, 5*mkt_return - 4* rf_rate], color = 'blue', label = 'CML')

plt.colorbar()
#plt.xlim(0.15,0.6)
#plt.ylim(0.15,0.6)
plt.xlabel('Risk')
plt.ylabel("Return")
plt.legend(loc=2, fontsize = 20)
plt.show
```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_17_1.png)
    





```python
#Portfolio beta

data_beta = pd.concat([data, mkt_data], axis = 1)

data_beta.columns = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

ret_beta = data_beta.pct_change().dropna()
mean_beta = ret_beta.mean() * 252
var_beta = ret_beta.var() * 252
std_beta = np.sqrt(var_beta)
cov_beta = ret_beta.cov()* 252

```


```python
beta = cov_beta[-1:]/cov_beta.iloc[-1,-1]
```


```python
def beta(ticker, start, end):
    data= web.get_data_yahoo(ticker, start, end)['Adj Close']
    market = web.get_data_yahoo('^sp1500', start, end)['Adj Close']
    data_beta = pd.concat([data, market], axis = 1)
    ret_beta = data_beta.pct_change().dropna()
    mean_beta = ret_beta.mean() * 252
    var_beta = ret_beta.var() * 252
    std_beta = np.sqrt(var_beta)
    cov_beta = ret_beta.cov()* 252
    beta = cov_beta[-1:]/((cov_beta.iloc[-1,-1]))
    beta.index = ['Beta']
    
    return beta, mean_beta, std_beta

```


```python
beta_asset, return_asset, std_asset =beta(ticker, '2017-07-01', end_date)
```


```python

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Beta</th>
      <td>27.902958</td>
      <td>25.655402</td>
      <td>27.40785</td>
      <td>23.634185</td>
      <td>33.289531</td>
      <td>23.397994</td>
    </tr>
  </tbody>
</table>
</div>




```python
def ex_ret (ticker, start_date, end_date):
    beta_asset, return_asset, std_asset =beta(ticker, start_date, end_date)
    ex_ret = rf_rate + beta_asset *(return_asset[-1] - rf_rate)
    ex_ret.index = ['expected_return']
    return(ex_ret)
```


```python
ex_ret(ticker, '2017-07-01', end_date)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>expected_return</th>
      <td>0.119066</td>
      <td>0.113362</td>
      <td>0.117809</td>
      <td>0.108232</td>
      <td>0.132738</td>
      <td>0.107632</td>
    </tr>
  </tbody>
</table>
</div>


