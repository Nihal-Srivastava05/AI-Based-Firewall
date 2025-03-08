import time
import numpy as np
import pickle
from datetime import datetime, timezone
import math

from re import compile
from urllib.parse import urlparse
from socket import gethostbyname
import warnings

import tensorflow as tf
from keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences

warnings.simplefilter("ignore", UserWarning)

feature_labels = ['benign','defacement', 'malware', 'phishing']

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
        data['entropy'] = self.entropy()
        data['numDigits'] = self.numDigits()
        data['urlLength'] = self.urlLength()
        data['numParams'] = self.numParameters()
        data['hasHttp'] = self.hasHttp()
        data['hasHttps'] = self.hasHttps()
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
        data['count_arate'] = self.count_arate()
        data['count_asterisk'] = self.count_asterisk()
        data['count_questionmark'] = self.count_questionmark()
        data['count_plus'] = self.count_plus()
        data['count_exclamation'] = self.count_exclamation()
        data['count_hyphen'] = self.count_hyphen()
        data['count_equal'] = self.count_equal()
        data['count_tilted'] = self.count_tilted()
    
        return data

firewall_clf = pickle.load(open("./Models/random_forest_base.pkl", 'rb'))
url_clf =  tf.keras.models.load_model("./Models/url_clsf_word_embed")

def get_padded_url(url, VOCAB_LENGTH = 464223, length_long_sentence = 373):
    one_hot_url = one_hot(url, VOCAB_LENGTH)
    padded_url = pad_sequences([one_hot_url], length_long_sentence, padding='post')
    return padded_url

print("########################## Model Loaded ##########################")

LAST_LINE = 0
junk_contents = ["[I]", '-', 'bytes', 'None']
while(True):
    time.sleep(1)
    f = open("./log_file.txt", "r")
    data = f.readlines()[LAST_LINE:]
    for d in data:
        content = d.split(' ')
        cleared_contents = [c for c in content if c not in junk_contents]

        flag = 0
        if(len(cleared_contents) < 9):
            continue
        if(len(cleared_contents) == 9):
            if(":" not in cleared_contents[4] or ":" not in cleared_contents[6]):
                continue
            url = cleared_contents[6]
            source_port = int(cleared_contents[4].split(":")[-1])
            dest_port = int(cleared_contents[6].split(":")[1].split("/")[0])
            byte = int(cleared_contents[7])
            elps_time = float(cleared_contents[8][:-3])
            firewall_clf_data = [source_port, dest_port, byte, 1, elps_time]
            flag = 1
        if(len(cleared_contents) == 11):
            if(":" not in cleared_contents[4] or ":" not in cleared_contents[6]):
                continue
            url = cleared_contents[6]
            source_port = int(cleared_contents[4].split(":")[-1])
            dest_port = int(cleared_contents[6].split(":")[1].split("/")[0])
            byte = int(cleared_contents[9])
            elps_time = float(cleared_contents[10][:-3])
            firewall_clf_data = [source_port, dest_port, byte, 1, elps_time]
            flag = 1

        if(flag == 1):
            instance1_data = np.array(firewall_clf_data)
            prediction1 = firewall_clf.predict([instance1_data])[0]
            url_feature_extraction = UrlFeaturizer(url).run()        
            url_feature_extraction = list(url_feature_extraction.values())
            instance2_data = np.array(url_feature_extraction)
            padded_url = get_padded_url(url)
            # prediction2 = url_clf.predict(x=[padded_url.reshape(1, 373), instance2_data.reshape(1, 28)], verbose=0)
            prediction2 = url_clf.predict(x=[padded_url.reshape(1, 373)], verbose=0)
            prediction2 = feature_labels[np.argmax(prediction2)]

            flag = 0
            if(prediction1 == 1):
                print(url, "Malicious --> Block")
                flag = 1

            if (prediction2 != 'benign'):
                print(url, "Malicious --> Block ", prediction2)
                flag = 1

            if(flag == 0):
                print(url, "Not Harmful --> Allow ", prediction2)

    LAST_LINE += len(data)