import requests
from datetime import datetime, timedelta

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
API_KEY_NEWS = '58c11d11a81a4ac59476e3da65b5874b'

class News:
    def __init__(self):
        pass

    def get_news(self, name, date_from, date_to,):
        parameters = {'q': name, 'apikey': API_KEY_NEWS, 'from': date_from, 'to': date_to, 'searchIn': 'title'}
        request = requests.get(NEWS_ENDPOINT, params=parameters)
        return request.json()['articles']

# now = datetime.now()
#
# date_from = now - timedelta(days = 1)
# date_from = date_from.strftime('%Y-%m-%d')
# date_to = now.strftime('%Y-%m-%d')
#
# news = News()
# print(date_from, date_to)
# articles = news.get_news('Ethereum', date_from, date_to)
#
# for article in articles:
#     print(article['publishedAt'])
#     print(datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'))
