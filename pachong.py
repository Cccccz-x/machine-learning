import re
import json
import requests
from bs4 import BeautifulSoup

response = requests.get('https://ncov.dxy.cn/ncovh5/view/pneumonia')#get信息
home_page = response.content.decode()#对信息解码
print(home_page)
soup = BeautifulSoup(home_page, 'lxml')#解析html文件
#print(soup)
script = soup.find(id='getAreaStat')#找到对应数据
#print(script)
json_str = re.findall(r'\[.+\]', script.text)[0]#找到具体数据
f = open('test.json', 'w')
f.write(json_str)
#rs = json.loads(json_str)#将json字符串转化为python字符串 文件用load不用s下同
#json_str = json.dumps(rs, ensure_ascii=False)#将python字符串转化为json字符串ascii转化为中文
#with open('test.json', 'w') as fp:
#    json.dump(rs, fp, ensure_ascii=False)

