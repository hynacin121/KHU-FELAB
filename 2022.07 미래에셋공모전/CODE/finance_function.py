from cmath import e
import numpy as np
import pandas as pd

import pandas_datareader as web
import cvxopt as opt
from cvxopt import solvers
from scipy import optimize as op


def dataread(ticker, start, end):
    data = web.get_data_yahoo(ticker, start, end)['Adj Close']
    ret = data.pct_change().dropna()

    return (data, ret)


def feasible_set(ret, count, rf_rate):
    n = len(ret.columns)
    returns = []
    stds = []
    wgt = []
    tr = 0

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


    wgt2 = pd.DataFrame(wgt, columns = ret.columns)
    sharpe = np.array(returns)/np.array(stds)
    tangent = (np.array(returns)- rf_rate)/np.array(stds)
    dt = {'Returns':returns, 'Stds': stds, 'Sharpe' : sharpe, 'Tangent' : tangent}
    dt = pd.DataFrame(dt)

    eff = pd.concat([dt,wgt2], axis = 1)

    return(eff)

def target_feasible(df, target):
    target_df = pd.DataFrame([(df.Returns - target)**2]).transpose()
    target_df.columns = ['LS']
    target_df = pd.concat([df, target_df], axis = 1)
    indexnum = target_df['LS'].idxmin()
    return (df.loc[[indexnum]])

def gmv_feasible(df):
    index = df['Stds'].idxmin()
    return (df.loc[[index]])
    

def gmv(x):
    return (np.dot(x.T, np.dot(e, x)))

def gmvportfolio(data): 
    global e 
    e = data.cov()*252
    x0 = [1/len(data.columns) for i in range(len(data.columns))]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, )  #가중치의 합= 1
    bound = (0.0, 1.0)
    bounds =tuple(bound for i in range(len(data.columns))) #Long only
    options = {'ftol' : 1e-20, }

    sol = op.minimize(gmv, x0, constraints = constraints, 
                    bounds = bounds, options = options, method='SLSQP',)
    return sol


def target_portfolio(data, target):
    global e 
    e = data.cov()*252
    x0 = [1/len(data.columns) for i in range(len(data.columns))]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
                    {"type" : 'eq', 'fun' : lambda x : np.sum(x * data.mean()) *252- target})  #가중치의 합= 1
    bound = (0.0, 1.0)
    bounds =tuple(bound for i in range(len(data.columns))) #Long only
    options = {'ftol' : 1e-20, }

    sol = op.minimize(gmv, x0, constraints = constraints, 
                    bounds = bounds, options = options, method='SLSQP',)
    return sol

def portfolio_result(sol, ret, rf_rate):
    tgt_wgt = sol.x.transpose()
    tgt_ret = np.sum(ret.mean() * tgt_wgt * 252)
    tgt_std = np.sqrt(sol.fun)
    tgt_sharp = tgt_ret/tgt_std
    tgt_tangent = (tgt_ret - rf_rate)/tgt_std
    tgt_wgt = pd.DataFrame(tgt_wgt).T
    tgt_wgt.columns = ret.columns
    dt = {"Returns" : tgt_ret, 'Stds' : tgt_std, 'Sharpe' : tgt_sharp, 'Tangent' : tgt_tangent}
    dt = pd.DataFrame(dt, index = [0])
    eff3 = pd.concat([dt, tgt_wgt], axis =1)
    return(eff3)

def compare_feas_opt(feas, opt):
    compare = pd.concat([feas, opt])
    compare.index = ['feas', 'opt']
    return compare