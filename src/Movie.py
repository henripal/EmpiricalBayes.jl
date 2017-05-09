import time
import random
import urllib.request
from bs4 import BeautifulSoup

class Movie():
    def __init__(self, url):
        url = url
        soup = self.make_soup_from_url()

    def make_soup_from_url(self, attempt=0):
        attempt = attempt + 1
        time.sleep(int(random.random()*20))
        if attempt > 10:
            return None
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'chrome'})
            html = urllib.request.urlopen(req).read()
            return BeautifulSoup(html, "lxml")
        except:
            self.make_soup_from_url(self, attempt)
