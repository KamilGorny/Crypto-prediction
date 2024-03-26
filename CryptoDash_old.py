# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:27:51 2022

@author: Luis
"""
import requests
import time
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pprint
import numpy as np

import dash
from dash import dcc, html
import plotly
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.models import load_model


class Krypto:
    def __init__(self, nazwa, kwota, zakup):
        self.nazwa = nazwa
        self.kwota = kwota
        self.zakup = zakup
        self.cena = []
        self.now = []
        self.ilosc = 0
        self.bid = []
        self.bidIlosc = []
        self.ask = []
        self.askIlosc = []
        self.cenaBuy = []
        self.cenaSell = []
        self.info = ''
        self.min = []
        self.max = []
        self.flag = 0
        self.df = pd.DataFrame()
        self.candleCloseT = []
        self.candleMin = []
        self.candleMax = []
        self.dzien = []
        self.candleClose = pd.DataFrame()
        self.scaler_x, self.scaler_y = pickle.load(open('scaler.pkl', 'rb'))
        self.model_2h = load_model('model_ETH_2h.h5', compile=False)
        self.model_5h = load_model('model_ETH_5h.h5', compile=False)
        self.x_data = []
        self.y_cena = []
        self.y_pred_2h = []
        self.y_pred_5h = []

    def getCrypto(self):
        url = "https://api.zonda.exchange/rest/trading/ticker"
        headers = {'content-type': 'application/json'}
        status = ''
        while status != 'Ok':
            response = requests.request("GET", url, headers=headers)
            status = json.loads(response.text)['status']
            if status != 'Ok':
                print(status)
                time.sleep(10)
        # self.cena=json.loads(response.text)['items'][self.nazwa+'-PLN']['rate']
        teraz = datetime.datetime.now()
        teraz = teraz.strftime("%Y-%m-%d %H:%M:%S")
        self.now.append(teraz)
        self.cena.append(float(json.loads(response.text)['items'][self.nazwa + '-PLN']['rate']))

        self.max.append(float(json.loads(response.text)['items'][self.nazwa + '-PLN']['highestBid']))
        self.min.append(float(json.loads(response.text)['items'][self.nazwa + '-PLN']['lowestAsk']))
        if len(self.cena) > 200:
            self.cena = self.cena[50:]
            self.max = self.max[50:]
            self.min = self.min[50:]
            self.now = self.now[50:]
        self.df = pd.DataFrame(self.cena, index=self.now, columns=['cena'])

    def getBid(self):
        headers = {'content-type': 'application/json'}
        url = "https://api.zonda.exchange/rest/trading/orderbook/" + self.nazwa + "-PLN"
        status = ''
        while status != 'Ok':
            response = requests.request("GET", url, headers=headers)
            status = json.loads(response.text)['status']
            if status != 'Ok':
                print(status)
                time.sleep(10)
        # response = requests.request("GET", url, headers=headers)
        # price=pprint.pprint(json.loads(response.text)['sell'][0:20])
        for i in range(50):
            self.bid.append(float(json.loads(response.text)['sell'][i]['ra']))
            self.bidIlosc.append(float(json.loads(response.text)['sell'][i]['ca']))
            self.ask.append(float(json.loads(response.text)['buy'][i]['ra']))
            self.askIlosc.append(float(json.loads(response.text)['buy'][i]['ca']))

        # print(response.text)

    def getCandlle(self):
        # o	decimal	Kurs otwarcia.
        # c	decimal	Kurs zamknięcia.
        # h	decimal	Najwyższa wartość kursu.
        # l	decimal	Najniższa wartość kursu.
        # v	decimal	Wygenerowany wolumen.
        self.df = pd.DataFrame()

        teraz = datetime.datetime.now()

        stop = teraz  # -datetime.timedelta(days=1)
        stop = stop.strftime("%d/%m/%Y %H:%M:%S")

        start = teraz - datetime.timedelta(hours=85 * 2)
        stop = datetime.datetime.now()  # start+relativedelta(months=6)

        stop = stop.strftime("%d/%m/%Y %H:%M:%S")
        start = start.strftime("%d/%m/%Y %H:%M:%S")

        start = datetime.datetime.strptime(start, "%d/%m/%Y %H:%M:%S").timestamp() * 1000
        stop = datetime.datetime.strptime(stop, "%d/%m/%Y %H:%M:%S").timestamp() * 1000
        start = str(start)[:-2]
        stop = str(stop)[:-2]
        nazwa = self.nazwa
        url = "https://api.zonda.exchange/rest/trading/candle/history/" + nazwa + "-PLN/3600?from=" + start + "&to=" + stop
        querystring = {"from": start, "to": stop}
        response = requests.request("GET", url, params=querystring)
        candle = json.loads(response.text)['items']
        close = []
        open = []
        candleMin = []
        candleMax = []
        dzien = []
        vol = []
        for i in candle:
            close.append(float(i[1]['c']))
            open.append(float(i[1]['o']))
            vol.append(float(i[1]['v']))
            # m=(float(i[1]['h'])+float(i[1]['l']))/2
            candleMin.append(float(i[1]['l']))
            candleMax.append(float(i[1]['h']))
            dzien.append(datetime.datetime.fromtimestamp(int(i[0]) / 1000))
        self.df = pd.DataFrame(
            {'data': dzien, 'cena': close, 'op': open, 'min': candleMin, 'max': candleMax, 'vol': vol})

        self.df['shortEMA'] = self.df['cena'].ewm(span=12, adjust=False).mean()
        # calculate long EMA
        self.df['longEMA'] = self.df['cena'].ewm(span=26, adjust=False).mean()
        # calculate MACD line
        self.df['MACD'] = self.df['shortEMA'] - self.df['longEMA']
        # calculate signal line
        self.df['signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        self.df['EMA_5'] = self.df['cena'].ewm(span=5, adjust=False).mean()
        self.df['WMA85'] = self.df['min'].rolling(85).apply(lambda x: x[::-1].cumsum().sum() * 2 / 85 / (85 + 1))
        self.df['WMA75'] = self.df['min'].rolling(75).apply(lambda x: x[::-1].cumsum().sum() * 2 / 75 / (75 + 1))
        self.df['signal_MACD'] = self.df['MACD'] - self.df['signal']
        self.df = self.df.dropna()

    def buy_sell(self):
        scaled_data_array = self.scaler_x.transform(self.df[['op', 'min', 'max', 'vol', 'shortEMA', 'longEMA',
                                                             'MACD', 'signal', 'EMA_5', 'WMA85', 'WMA75',
                                                             'signal_MACD']])
        scaled_data_array = scaled_data_array[-48:]
        X_ = []
        for i in range(24, len(scaled_data_array)):
            X_.append(scaled_data_array[i - 24:i, :])
        X_ = np.array(X_)

        self.y_pred_2h = self.scaler_y.inverse_transform(self.model_2h.predict(X_))
        self.y_pred_5h = self.scaler_y.inverse_transform(self.model_5h.predict(X_))
        y_pred_2h = self.y_pred_2h
        y_pred_5h = self.y_pred_5h
        self.getBid()
        Kup = (y_pred_2h[-1] > y_pred_2h[-2]) and (y_pred_5h[-1] > y_pred_5h[-2]) and (self.flag == 0) and (
                    self.kwota > 0)
        Sprzedaj = (y_pred_2h[-1] < y_pred_2h[-2]) and (y_pred_5h[-1] < y_pred_5h[-2]) and (self.flag > 0) and (
                    self.ask[0] > (self.cenaBuy + self.cenaBuy * 0.01)) and (self.ilosc > 0)

        if Sprzedaj == True:
            self.flag = 0
            kwotaTemp = self.zakup
            zarobek = 0
            ilosc = self.ilosc
            infoTemp = ''
            suma = 0
            i = 0
            while ilosc > 0:
                ilosc -= self.askIlosc[i]
                if ilosc < 0:
                    ilosc += self.askIlosc[i]
                    zarobek += ilosc * self.ask[i]
                    ilosc = 0
                else:
                    zarobek += self.askIlosc[i] * self.ask[i]
                i += 1
            self.info = infoTemp
            self.kwota = zarobek
            self.ilosc = 0
            self.cenaSell = self.ask[i]
            self.y_cena.append(self.ask[i])
            self.x_data.apend(datetime.datetime.now())

        if Kup == True:
            self.flag = 1
            self.info = ''
            kwota = self.kwota
            ilosc = 0
            teraz = str(datetime.datetime.now())
            suma = 0
            while kwota > 0:
                kwota -= self.bid[i] * self.bidIlosc[i]
                if kwota < 0:
                    kwota += self.bid[i] * self.bidIlosc[i]
                    ilosc += kwota / self.bid[i]
                    kwota = 0
                else:
                    ilosc += self.bidIlosc[i]
                i += 1
            self.ilosc = ilosc
            self.zakup = self.kwota
            self.kwota = 0
            self.cenaBuy = self.bid[i]
            self.y_cena.append(self.bid[i])
            self.x_data.append(datetime.datetime.now())


ETH = Krypto('ETH', 100, 0)
# XRP=Krypto('XRP', 1000,0)
# ADA=Krypto('ADA', 1000,0)
# BAT=Krypto('BAT', 1000,0)
# LUNA=Krypto('LUNA', 1000,0)
# DOT=Krypto('DOT', 1000,0)
T = [ETH]
# T=[ETH, XRP, ADA, BAT, LUNA, DOT]

teraz = []
teraz.append(datetime.datetime.now())
no = 0
N = []
for i in T:
    i.getCandlle()
    N.append(i.nazwa)
    # i.getCandlle()

app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        html.H4('Krypto'),
        html.Div(id='live-update-text'),
        dcc.RadioItems(N, N[0], id='crossfilter-xaxis-column',
                       labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
        dcc.Graph(id='live-update-graph'),
        html.Div(id='status-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=5 * 60 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    h = []
    style = {'padding': '5px', 'fontSize': '16px'}
    for i in T:
        i.getCandlle()
        i.buy_sell()
        hT = html.Span(['Nazwa: ' + i.nazwa], style=style)
        h.append(hT)
        # hT=html.Span(['Cena: ' + str(i.candleClose['cena'][-1])], style=style)
        # hT=html.Span(['Cena: ' + str(i.candleCloseT[-1])], style=style)
        # h.append(hT)
        # hT=html.Span(['Zakup: ' + str(i.zakup)], style=style)
        # h.append(hT)
        hT = html.Span(['Ilosc: ' + str(i.ilosc)], style=style)
        h.append(hT)
        hT = html.Span(['Kwota: ' + str(i.kwota)], style=style)
        h.append(hT)
        hT = html.Span(['Buy: ' + str(i.cenaBuy)], style=style)
        h.append(hT)
        hT = html.Span(['Sell: ' + str(i.cenaSell)], style=style)
        h.append(hT)
        hT = html.Span(['Interval: ' + str(n)], style=style)
        h.append(hT)

        h.append(html.Br())
        # if len(i.df['cena'])>85:
        # i.buy_sell()

    return h


# [
#        html.Span(['Cena: ' + str(ETH.cena[-1])], style=style),
#        html.Span(['Max: ' + str(ETH.max[-1])], style=style),
#        html.Span(['Min: ' + str(ETH.min[-1])], style=style)
#   ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'),
              Input('crossfilter-xaxis-column', 'value'))
def update_graph_live(n, nazwa):
    krypto = N.index(nazwa)
    l = 1
    r = len(T)
    fig = plotly.tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, horizontal_spacing=0.02)
    # teraz.append(datetime.datetime.now())
    i = T[krypto]
    i.getCandlle()
    i.buy_sell()
    # print('pred', list(i.df['data'][len(i.df['data']) - len(i.y_pred_5h):]), i.y_pred_2h[:, 0])
    fig.append_trace({
        'x': list(i.df['data']),
        'y': i.df['cena'],
        'name': i.nazwa,
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': i.x_data,
        'y': i.y_cena,
        'text': 'Zakup',
        'name': 'Zakup',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': i.df['data'],
        'y': i.df['EMA_5'],
        'text': 'EMA',
        'name': 'EMA_5',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': i.df['data'],
        'y': i.df['WMA85'],
        'text': 'WMA85',
        'name': 'WMA85',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': list(i.df['data'][len(i.df['data']) - len(i.y_pred_2h):]),
        'y': list(i.y_pred_2h[:, 0]),
        'text': '2h',
        'name': '2h',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': list(i.df['data'][len(i.df['data']) - len(i.y_pred_5h):]),
        'y': list(i.y_pred_5h[:, 0]),
        'text': '5h',
        'name': '5h',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    # fig.append_trace({
    #     'x': list(EMA_5.index),
    #     'y': EMA_5['cena'],
    #     'text': 'EMA',
    #     'name': 'EMA_5',
    #     'mode': 'lines',
    #     'type': 'scatter'
    #     }, 1, 1)
    # fig.append_trace({
    #     'x': list(EMA_5.index),
    #     'y': W85,
    #     'text': 'W85',
    #     'name': 'WMA_85',
    #     'mode': 'lines',
    #     'type': 'scatter'
    #     }, 1, 1)
    # fig.append_trace({
    #     'x': list(EMA_5.index),
    #     'y': W75,
    #     'text': 'W75',
    #     'name': 'WMA_75',
    #     'mode': 'lines',
    #     'type': 'scatter'
    #     }, 1, 1)
    #
    # fig.append_trace({
    #     'x': list(MACD.index),
    #     'y': MACD['cena'],
    #     'text': 'MACD',
    #     'name': 'MACD',
    #     'mode': 'lines',
    #     'type': 'scatter'
    #     }, 2, 1)
    # fig.append_trace({
    #     'x': list(signal.index),
    #     'y': signal['cena'],
    #     'text': 'signal',
    #     'name': 'signal',
    #     'mode': 'lines',
    #     'type': 'scatter'
    #     }, 2, 1)
    # fig.append_trace({
    #    'x': list(signal.index),
    #    'y': SYGNAL,
    #    'text': 'histogram',
    #    'name': 'histogram',
    #    'mode': 'lines',
    #    'type': 'scatter'
    #    }, l, 2)
    # fig.append_trace(go.Bar(x=list(signal.index), y=SYGNAL), 2, 1)
    # fig.append_trace(go.Bar(x=list(signal.index), y=SYGNAL), 2, 1)
    l += 1
    fig.update_layout(height=900, width=1700)
    return fig


def prepare_data_future(train_data, y_data, n=0):
    # Podział danych na X i y:
    X_ = []
    y_ = []

    for i in range(24, len(train_data) - n):
        X_.append(train_data[i - 24:i, :])
        # y_.append(train_data[i + n, 0])
        y_.append(y_data[i + n, 0])
        # X_.append(train_data[i-24:i])
        # y_.append(train_data[i + n])

    print(len(X_), len(y_))
    X_ = np.array(X_)
    print(X_.shape)
    # X_ = np.reshape(X_, (X_.shape[0], X_.shape[1], 1))
    return X_, np.array(y_)


if __name__ == '__main__':
    # ETH = Krypto('ETH', 100, 0)
    # ETH.getCandlle()
    # ETH.getBid()
    # # #print(ETH.df.dropna())
    # scaler_x,scaler_y = pickle.load(open('scaler.pkl', 'rb'))
    # scaled_data_array = scaler_x.transform(ETH.df[['op', 'min', 'max', 'vol', 'shortEMA', 'longEMA',
    #                                               'MACD', 'signal', 'EMA_5', 'WMA85', 'WMA75', 'signal_MACD']])
    # #
    # scaled_data_array = scaled_data_array[-48:]
    # X_ =[]
    # for i in range(24, len(scaled_data_array)):
    #     X_.append(scaled_data_array[i - 24:i, :])
    # X_ = np.array(X_)
    # print(X_.shape)
    #
    # model_2h = load_model('model_ETH_2h.h5')
    # model_5h = load_model('model_ETH_5h.h5')
    # y_pred_2h = scaler_y.inverse_transform(model_2h.predict(X_))
    # y_pred_5h = scaler_y.inverse_transform(model_5h.predict(X_))
    # # print(y_pred_2h)
    # print(len(y_pred_5h[:, 0]))
    # print(len(y_pred_2h))
    # print(len(ETH.df['data'])-len(ETH.y_pred_2h))
    # print(ETH.df['data'][len(ETH.df['data'])-len(ETH.y_pred_2h):])
    # print(ETH.bid)
    # print(ETH.bid[0])
    # print(ETH.ask)
    # print(ETH.ask[0])

    app.run_server(debug=True)
