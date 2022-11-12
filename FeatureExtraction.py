import pandas as pd
from datetime import datetime, timezone
import math
import warnings

from re import compile
from urllib.parse import urlparse
from socket import gethostbyname

warnings.simplefilter(action='ignore', category=FutureWarning)

class UrlFeaturizer(object):
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.urlparse = urlparse(url)

    def entropy(self):
        string = self.url.lower().strip()
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

    def hasHttp(self):
        return 'http:' in self.url

    def hasHttps(self):
        return 'https:' in self.url

    def url_host_is_ip(self):
        host = self.urlparse.netloc
        pattern = compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        match = pattern.match(host)
        return match is not None

    def get_ip(self):
        try:
            ip = self.urlparse.netloc if self.url_host_is_ip() else gethostbyname(self.urlparse.netloc)
            return ip
        except:
            return None

    def url_path_length(self):
        return len(self.urlparse.path)

    def url_host_length(self):
        return len(self.urlparse.netloc)

    def url_has_port_in_string(self):
        has_port = self.urlparse.netloc.split(':')
        return len(has_port) > 1 and has_port[-1].isdigit()

    def is_encoded(self):
        return '%' in self.url.lower()

    def num_encoded_char(self):
        encs = [i for i in self.url if i == '%']
        return len(encs)

    def number_of_subdirectories(self):
        d = self.urlparse.path.split('/')
        return len(d)

    def number_of_periods(self):
        periods = [i for i in self.url if i == '.']
        return len(periods)

    def has_client_in_string(self):
        return 'client' in self.url.lower()

    def has_admin_in_string(self):
        return 'admin' in self.url.lower()

    def has_server_in_string(self):
        return 'server' in self.url.lower()

    def has_login_in_string(self):
        return 'login' in self.url.lower()
        
    def get_tld(self):
        #top-level domain
        return self.urlparse.netloc.split('.')[-1].split(':')[0]

    def count_arate(self):
        arates = [i for i in self.url if i == '@']
        return len(arates)

    def count_asterisk(self):
        asterisks = [i for i in self.url if i == '*']
        return len(asterisks)

    def count_questionmark(self):
        questionmarks = [i for i in self.url if i == '?']
        return len(questionmarks)

    def count_plus(self):
        plus = [i for i in self.url if i == '+']
        return len(plus)

    def count_exclamation(self):
        exclamation = [i for i in self.url if i == '!']
        return len(exclamation)

    def count_hyphen(self):
        hyphen = [i for i in self.url if i == '-']
        return len(hyphen)

    def count_equal(self):
        equals = [i for i in self.url if i == '=']
        return len(equals)

    def count_tilted(self):
        tilted = [i for i in self.url if i == '~']
        return len(tilted)

    def run(self):
        data = {}
        data['url'] = self.url
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
        data['path_length'] = self.url_path_length()
        data['host_length'] = self.url_host_length()
        data['has_port'] = self.url_has_port_in_string()
        data['is_encoded'] = self.is_encoded()
        data['num_encoded_char'] = self.num_encoded_char()
        data['number_of_subdirectories'] = self.number_of_subdirectories()
        data['number_of_periods'] = self.number_of_periods()
        data['has_client_in_string'] = self.has_client_in_string()
        data['has_admin_in_string'] = self.has_admin_in_string()
        data['has_server_in_string'] = self.has_server_in_string()
        data['has_login_in_string'] = self.has_login_in_string()
        data['tld'] = self.get_tld()
        data['count_arate'] = self.count_arate()
        data['count_asterisk'] = self.count_asterisk()
        data['count_questionmark'] = self.count_questionmark()
        data['count_plus'] = self.count_plus()
        data['count_exclamation'] = self.count_exclamation()
        data['count_hyphen'] = self.count_hyphen()
        data['count_equal'] = self.count_equal()
        data['count_tilted'] = self.count_tilted()
    
        return data

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
    df = pd.read_csv('./Datasets/malicious_phish.csv')

    with Manager() as manager:
        FEATURE_DFS = manager.list()
        processes = []
        idx = 15000
        start_idx = 650001
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