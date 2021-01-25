#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import random
import json
from bs4 import BeautifulSoup

user_agent_file = open("user_agents.txt", "r")
user_agents = [str(i.strip()) for i in user_agent_file.readlines()]
user_agent = random.choice(user_agents)


headers = {
        'user-agent':user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Pragma': 'no-cache'
           }

url_to_title = {}
count = 0

def get_urls_to_titles_on_page(index_url):
    r = requests.get(url, headers=headers, )
    
    if r.status_code == 200:
        r_text = r.text
        
        soup = BeautifulSoup(r_text, 'html.parser')
        
        page_list = soup.find("div",id="mw-pages")
        
        categories = page_list.find_all("div",class_="mw-category-group")
        
        for category in categories:
            all_a = category.find_all("a")
            for link in all_a:
                url_to_title[link["href"]] = link.text
        
        next_link = page_list.find_all("a")
        link = [i for i in next_link if i.text == "next page"]
        global count
        print(count)
        count += 1 
        return "https://allthetropes.org" + link[0]["href"]
    else:
        print("error")
        return False
        

url = "https://allthetropes.org/w/index.php?title=Category:Trope"

while url:
    url = get_urls_to_titles_on_page(url)

with open("tropes_url_to_title.json", "w") as write_file:
    json.dump(url_to_title, write_file)
    
