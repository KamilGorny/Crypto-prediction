import json
from keras.models import load_model
import pickle
from datetime import datetime, timedelta
import requests
import time
import pandas as pd
import numpy as np

URL = "https://api.zonda.exchange/rest/trading/ticker"
HEADERS = {'content-type': 'application/json'}
URL_CANDLE = "https://api.zonda.exchange/rest/trading/candle/history/"
URL_ORDERBOOK = "https://api.zonda.exchange/rest/trading/orderbook/"

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
API_KEY_NEWS = '58c11d11a81a4ac59476e3da65b5874b'


class Krypto:
    def __init__(self, crypto_name: str, cash: float, quantity: float, currency: str):
        self.crypto_name = crypto_name
        self.cash = cash
        self.cena = 0
        self.quantity = quantity
        self.currency = currency
        self.bid = []
        self.bid_quantity = []
        self.ask = []
        self.ask_quantity = []
        self.buy_price = 0
        self.sell_price = 0
        self.df = pd.DataFrame()
        self.actions = []
        self.scaler_x, self.scaler_y = pickle.load(open(f'scaler_{crypto_name}.pkl', 'rb'))
        self.model_2h = load_model(f'model_2h_{crypto_name}.h5', compile=False)
        self.model_5h = load_model(f'model_5h_{crypto_name}.h5', compile=False)
        self.y_pred_2h = []
        self.y_pred_5h = []
        #self.articles = pd.DataFrame(columns=['publishedAt', 'title', 'content', 'url'])
        self.articles = pd.DataFrame(columns=['publishedAt', 'title', 'content'])

    def response(self, url, header):
        status = ''
        while status != 'Ok':
            response = requests.get(url, headers=header)
            status = json.loads(response.text)['status']
            if status != 'Ok':
                print(status)
                time.sleep(10)
        return response

    def getCrypto(self):
        response = self.response(URL, HEADERS)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f'{self.crypto_name}-{self.currency}'
        self.now.append(now)
        self.cena.append(float(json.loads(response.text)['items'][name]['rate']))

        self.max.append(float(json.loads(response.text)['items'][name]['highestBid']))
        self.min.append(float(json.loads(response.text)['items'][name]['lowestAsk']))
        if len(self.cena) > 200:
            self.cena = self.cena[50:]
            self.max = self.max[50:]
            self.min = self.min[50:]
            self.now = self.now[50:]
        self.df = pd.DataFrame(self.cena, index=self.now, columns=['cena'])

    def getBid(self):
        url = f'{URL_ORDERBOOK}{self.crypto_name}-{self.currency}'
        response = self.response(url, HEADERS)

        for i in range(50):
            self.bid.append(float(json.loads(response.text)['sell'][i]['ra']))
            self.bid_quantity.append(float(json.loads(response.text)['sell'][i]['ca']))
            self.ask.append(float(json.loads(response.text)['buy'][i]['ra']))
            self.ask_quantity.append(float(json.loads(response.text)['buy'][i]['ca']))

    def getCandlle(self):
        # o	decimal	Kurs otwarcia.
        # c	decimal	Kurs zamknięcia.
        # h	decimal	Najwyższa wartość kursu.
        # l	decimal	Najniższa wartość kursu.
        # v	decimal	Wygenerowany wolumen.
        self.df = pd.DataFrame()

        now = datetime.now()

        stop = now.strftime("%d/%m/%Y %H:%M:%S")

        start = now - timedelta(hours=85 * 2)
        start = start.strftime("%d/%m/%Y %H:%M:%S")

        start = datetime.strptime(start, "%d/%m/%Y %H:%M:%S").timestamp() * 1000
        stop = datetime.strptime(stop, "%d/%m/%Y %H:%M:%S").timestamp() * 1000
        start = str(start)[:-2]
        stop = str(stop)[:-2]

        url = f'{URL_CANDLE}{self.crypto_name}-{self.currency}/3600?from={start}&to={stop}'
        querystring = {"from": start, "to": stop}
        response = self.response(url, querystring)

        response = response.json()['items']
        close_price = []
        open_price = []
        price_max = []
        price_min = []
        day = []
        vol = []
        for data in response:
            close_price.append(float(data[1]['c']))
            open_price.append(float(data[1]['o']))
            vol.append(float(data[1]['v']))
            price_min.append(float(data[1]['l']))
            price_max.append(float(data[1]['h']))
            day.append(datetime.fromtimestamp(int(data[0]) / 1000))
        self.df = pd.DataFrame(
            {'data': day, 'cena': close_price, 'op': open_price, 'min': price_min, 'max': price_max, 'vol': vol})

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

    def prediction(self):
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
        if (y_pred_2h[-1] > y_pred_2h[-2] * 1.005) and (y_pred_5h[-1] > y_pred_5h[-2] * 1.005):
            self.buy()
        if (y_pred_2h[-1] * 1.001 < y_pred_2h[-2]) and (y_pred_5h[-1] * 1.001 < y_pred_5h[-2]) and (
                self.ask[0] > self.buy_price):
            self.sell()
        return None

    def buy(self):
        if self.cash > 0:
            ind = 0
            while self.cash > 0:
                if self.cash < (self.bid[ind] * self.bid_quantity[ind]):
                    self.quantity += self.cash / self.bid[ind]
                    self.cash = 0
                else:
                    self.quantity += self.bid_quantity[ind]
                    self.cash -= self.bid[ind] * self.bid_quantity[ind]
                ind += 1
            self.buy_price = self.ask[0]
            self.actions.append = {'price': self.ask[0],
                                   'action': 'BUY',
                                   'date': datetime.now()}

    def sell(self):
        if self.quantity > 0:
            ind = 0
            while self.quantity > 0:
                if self.quantity < self.ask_quantity[ind]:
                    self.cash += self.quantity * self.ask[ind]
                    self.quantity = 0
                else:
                    self.quantity -= self.ask_quantity[ind]
                    self.cash += self.ask_quantity[ind] * self.ask[ind]
                ind += 1
            self.sell_price = self.bid[0]
            self.actions.append = {'price': self.bid[0],
                                   'action': 'SELL',
                                   'date': datetime.now()}

    def get_news(self):
        crypto_dict = {'ETH': 'Ethereum '}
        date_from = self.df['data'].max() - timedelta(days=1)
        date_to = self.df['data'].max() + timedelta(days=1)
        date_from = date_from.strftime('%Y-%m-%d')
        date_to = date_to.strftime('%Y-%m-%d')

        df = self.df[(self.df['data'] >= date_from) & (self.df['data'] <= date_to)]
        percentage = 100 - df['cena'].min() / df['cena'].max() * 100
        if abs(percentage) >= 0:
            self.articles = pd.DataFrame(columns=['publishedAt', 'title', 'content'])
            parameters = {'q': crypto_dict[self.crypto_name],
                          'apikey': API_KEY_NEWS,
                          'from': date_from,
                          'to': date_to,
                          'searchIn': 'title'}
            request = requests.get(NEWS_ENDPOINT, params=parameters)
            articles = {}
            for article in request.json()['articles']:
                articles['publishedAt'] = datetime.strptime(article['publishedAt'].split("T")[0], '%Y-%m-%d')
                articles['title'] = article['title']
                articles['description'] = article['description']
                articles['url'] = article['url']
                #row = [datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                #       article['title'],
                #       article['description'],
                #       article['url']]
                row = [datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                       '['+article['title']+']'+'('+article['url']+')',
                       article['description']]
                self.articles.loc[len(self.articles)] = row
            # self.articles.append(articles, ignore_index=True)
            self.articles = self.articles[self.articles['publishedAt'] >= self.df['data'].min()]
            self.articles = self.articles.sort_values(by='publishedAt', ascending=False)

