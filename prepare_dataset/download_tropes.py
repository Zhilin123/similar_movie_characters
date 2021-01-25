#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import random
from bs4 import BeautifulSoup
import re
import string
import json


user_agent_file = open("user_agents.txt", "r")
user_agents = [str(i.strip()) for i in user_agent_file.readlines()]

user_agent = user_agents[0]
headers = {
        'user-agent':user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Pragma': 'no-cache'
           }

with open("tropes_url_to_title.json", "r") as read_file:
     url_to_title = json.load(read_file)
     

def get_tropes_from_one_url(url):

    r = requests.get(url, headers=headers, )
    
    r_text = r.text
            
    soup = BeautifulSoup(r_text, 'html.parser')
    
    all_stories = []
    
    categories = soup.find_all("div",class_="hs-block")
    
    for category in categories:
        try:
            h2 = category.find("h2")
            if h2 == None:
                h2 = category.find("h3")
            if h2 == None:
                h2 = category.find("h4")
            if h2 == None:
                h2 = category.find("h5")
            if h2 == None:
                h2 = category.find("h1")
            story_type = h2.find("span", class_="mw-headline").text
            all_posts = category.find("div", class_="hs-section")
            all_uls = all_posts.findChildren("ul", recursive=False)
            
            for ul in all_uls:
                all_li_children = ul.findChildren("li", recursive=False)
                for li in all_li_children:
                    text_messy = re.sub('<[^<]+?>', '', str(li))
                    text_order = ' '.join(text_messy.split())
                    one_story = {
                            'story_type':story_type,
                            'text':text_order
                            }
                    all_stories.append(one_story)
        except:
            print(url)
            print(category)
    return all_stories

url_keys = list(url_to_title.keys())

url_to_tropes = {}

for i in range(len(url_keys)):
    url = url_keys[i]
    tropes = get_tropes_from_one_url("https://allthetropes.org"+url)
    url_to_tropes[url] = tropes
    if i % 10 == 0:
        print(i)
    if i % 2000 == 0 and i != 0:
        with open("%d_tropes_url_to_tropes.json"%(i), "w") as write_file:
            json.dump(url_to_tropes, write_file)
        url_to_tropes = {}
    count = i

with open("%d_tropes_url_to_tropes.json"%(count), "w") as write_file:
    json.dump(url_to_tropes, write_file)

    