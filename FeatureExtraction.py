import pandas as pd
from datetime import datetime, timezone
import math
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

class UrlFeaturizer(object):
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]

    def entropy(self):
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def ip(self):
        string = self.url
        flag = False
        if ("." in string):
            elements_array = string.strip().split(".")
            if(len(elements_array) == 4):
                for i in elements_array:
                    if (i.isnumeric() and int(i)>=0 and int(i)<=255):
                        flag=True
                    else:
                        flag=False
                        break
        if flag:
            return 1 
        else:
            return 0

    def numDigits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    def urlLength(self):
        return len(self.url)

    def numParameters(self):
        params = self.url.split('&')
        return len(params) - 1

    def numFragments(self):
        fragments = self.url.split('#')
        return len(fragments) - 1

    def numSubDomains(self):
        subdomains = self.url.split('http')[-1].split('//')[-1].split('/')
        return len(subdomains)-1

    def domainExtension(self):
        ext = self.url.split('.')[-1].split('/')[0]
        return ext

    ## URL domain features
    def hasHttp(self):
        return 'http:' in self.url

    def hasHttps(self):
        return 'https:' in self.url

    def run(self):
        data = {}
        data['entropy'] = self.entropy()
        data['numDigits'] = self.numDigits()
        data['urlLength'] = self.urlLength()
        data['numParams'] = self.numParameters()
        data['hasHttp'] = self.hasHttp()
        data['hasHttps'] = self.hasHttps()
        data['ext'] = self.domainExtension()
        data['num_%20'] = self.url.count("%20")
        data['num_@'] = self.url.count("@")
        data['has_ip'] = self.ip()
    
        return data

df = pd.read_csv('./Datasets/malicious_phish.csv')

from multiprocessing import Process, Manager
import time
import os

def feature_extraction(dataset, FEATURE_DFS):
    print("PID {} Started".format(os.getpid()))
    feature_df = pd.DataFrame()
    for row in dataset.iterrows():
        temp = UrlFeaturizer(row[1].url).run()
        temp['target'] = row[1].type
        feature_df = feature_df.append(temp, ignore_index=True)
        if(row[0] % 5000 == 0):
            print("PID: {} at {}".format(os.getpid(), row[0]))
    FEATURE_DFS.append(feature_df)
    print("DONE {}".format(os.getpid()))


if __name__ == "__main__":
	with Manager() as manager:
		FEATURE_DFS = manager.list()
		processes = []
		idx = 10000
		start_idx = 630002
		for i in range(1, 15):
			p = Process(target=feature_extraction, args=(df[start_idx+idx*(i-1):start_idx+idx*i], FEATURE_DFS))
			p.start()
			processes.append(p)

		for p in processes:
			p.join()
		print("DONE!")

		final_df = pd.concat(FEATURE_DFS)
		final_df.reset_index()
		final_df.to_csv('./Datasets/feature_extraction3.csv', index=False)