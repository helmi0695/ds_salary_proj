# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:11:21 2021

@author: HB6
"""


import requests
from data_input import data_in


URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

r = requests.get(URL,headers=headers, json=data) 

r.json()