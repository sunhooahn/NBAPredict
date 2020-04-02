import numpy as np
import sklearn
from selenium import webdriver
from bs4 import BeautifulSoup as soup 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os
from webdriver_manager.chrome import ChromeDriverManager


def train():
    x = np.genfromtxt("nbastats.csv", delimiter = ",")
    y = np.genfromtxt("target.csv", delimiter = "\n")
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3, random_state =1)

    lin = linear_model.LinearRegression()
    lin.fit(xtrain,ytrain)
    return lin


def scraper():
   
    d  = webdriver.Chrome(ChromeDriverManager().install())
    d.get('https://stats.nba.com/teams/advanced/?sort=W&dir=-1')
    s = soup(d.page_source, 'html.parser').find('table')
    headers, [_, *data] = [i.text for i in s.find_all('th')], [[i.text for i in b.find_all('td')] for b in s.find_all('tr')]
    final_data = [i for i in data if len(i) > 1]


    parse = []
    names = []
    
    for i in final_data:
        parse.append(i[6:])
        names.append(i[1].strip())
    
    
    
    for i in parse:
        for j in range(0,14):
            i[j] = float(i[j])

    return parse, names


def display():
    
    sample, names = scraper()

    pred = train().predict(sample)

    ret = ""
    for i in range(0,30):
        ret = ret + (names[i] + ": " + str(pred[i]) + " wins") + "\n"
    return ret


    

print(display())