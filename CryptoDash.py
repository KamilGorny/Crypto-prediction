# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:27:51 2022

@author: Luis
"""

import datetime
import numpy as np

import dash
from dash import dcc, html, dash_table
import plotly
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from settings import *

no = 0
N = []
children = []
for i in CRYPTOCURRENCY:
    i.getCandlle()
    N.append(i.crypto_name)
    # i.getCandlle()
    children.append(dcc.Tab(label=i.crypto_name, value=i.crypto_name))

app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        html.H4('Krypto'),
        html.Div(id='live-update-text'),
        dcc.RadioItems(N, N[0], id='crossfilter-xaxis-column',
                       labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
        dcc.Tabs(id='tabs-example-1', value=N[0], children=children),
        dcc.Graph(id='live-update-graph'),
        html.Div(id='status-update-text'),
        dash_table.DataTable(id='news-table',
                             style_data={
                                 'whiteSpace': 'normal',
                                 'height': 'auto',
                             }),
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
    crypto_label = []
    style = {'padding': '5px', 'fontSize': '16px'}
    for i in CRYPTOCURRENCY:
        i.getCandlle()
        i.getBid()
        i.prediction()
        label_text = html.Span(['Nazwa: ' + i.crypto_name], style=style)
        crypto_label.append(label_text)
        # hT=html.Span(['Cena: ' + str(i.candleClose['cena'][-1])], style=style)
        # hT=html.Span(['Cena: ' + str(i.candleCloseT[-1])], style=style)
        # h.append(hT)
        # hT=html.Span(['Zakup: ' + str(i.zakup)], style=style)
        # h.append(hT)
        label_text = html.Span(['Ilosc: ' + str(i.quantity)], style=style)
        crypto_label.append(label_text)
        label_text = html.Span(['Kwota: ' + str(i.cash)], style=style)
        crypto_label.append(label_text)
        label_text = html.Span(['Buy: ' + str(i.buy_price)], style=style)
        crypto_label.append(label_text)
        label_text = html.Span(['Sell: ' + str(i.sell_price)], style=style)
        crypto_label.append(label_text)
        label_text = html.Span(['Interval: ' + str(n)], style=style)
        crypto_label.append(label_text)

        crypto_label.append(html.Br())

    return crypto_label

@app.callback(Output('live-update-graph', 'figure'),
              Output('news-table', 'data'),
              Output('news-table', 'columns'),
              Input('live-update-text', 'children'),
              Input('crossfilter-xaxis-column', 'value'))
def update_graph_live(n, name):
    ind = N.index(name)
    # fig = plotly.tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, horizontal_spacing=0.02)
    fig = go.Figure()
    crypto = CRYPTOCURRENCY[ind]
    x_data = crypto.df['data']
    fig.add_trace(go.Scatter(x=x_data,
                             y=crypto.df['cena'],
                             name=crypto.crypto_name,
                             mode='lines',
                             type='scatter'))
    fig.add_trace(go.Scatter(x=x_data,
                             y=crypto.df['EMA_5'],
                             text="EMA", name='EMA_5',
                             mode='lines',
                             type='scatter'))
    fig.add_trace(go.Scatter(x=x_data,
                             y=crypto.df['WMA85'],
                             text='WMA85',
                             name='WMA85',
                             mode='lines',
                             type='scatter'))
    fig.add_trace(go.Scatter(x=x_data,
                             y=crypto.df['WMA75'],
                             text='WMA75',
                             name='WMA75',
                             mode='lines',
                             type='scatter'))
    x = list(crypto.df['data'][len(crypto.df['data']) - len(crypto.y_pred_2h):])
    # x = [t + pd.Timedelta(hours=2) for t in x]
    #print(list(crypto.y_pred_2h[:, 0]))
    fig.add_trace(go.Scatter(x=x,
                             y=list(crypto.y_pred_2h[:, 0]),
                             text='2h',
                             name='2h',
                             mode='lines',
                             type='scatter'))
    x = list(crypto.df['data'][len(crypto.df['data']) - len(crypto.y_pred_5h):])
    #x = [t + pd.Timedelta(hours=5) for t in x]
    fig.add_trace(go.Scatter(x=x,
                             y=list(crypto.y_pred_5h[:, 0]),
                             text='5h',
                             name='5h',
                             mode='lines',
                             type='scatter'))

    for action in crypto.actions:
        fig.add_annotation(x=action['data'],
                           y=action['price'],
                           text=f'{action["action"]} for {action["price"]}',
                           showarrow=True,
                           arrowhead=1)
    crypto.get_news()
    table_data = None
    if len(crypto.articles) > 0:
        table_data = crypto.articles.to_dict(orient='records')
        columns = [{"name": "publishedAt", "id": "publishedAt"},
                   {"name": "title", "id": "title", 'type': 'text', 'presentation': 'markdown'},
                   {"name": "content", "id": "content"}]

    fig.update_layout(height=500, width=1700)

    return fig, table_data, columns


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
    # from datetime import datetime, timedelta
    #
    #ETH = Krypto('ETH', 100, 0, 'PLN')
    # #
    #ETH.getCandlle()
    #ETH.prediction()
    #print(list(ETH.y_pred_2h[:, 0]))
    #print(ETH.df)
    # now = datetime.now()
    # d1 = datetime(now.year, now.month, now.day)
    # d2 = d1 - timedelta(days=1)
    # df = ETH.df[(ETH.df['data'] >= d2) & (ETH.df['data'] <= d1)]
    # print(df)
    # r = 100 - df['cena'].min()/df['cena'].max()*100
    # print(r)
    # ETH.getBid()
    # ETH.buy()
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
