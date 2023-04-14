from urllib.parse import urlparse, urlsplit, parse_qs
import tldextract
import math
import string
from collections import Counter
import re
import requests
import numpy as np
import pandas as pd

def url_components(url):
    """Returns the subdomain, domain, hostname, TLD, base URL, pathway, and scheme of a URL."""
    parsed_url = urlparse(url)
    split_url = urlsplit(url)
    scheme = split_url.scheme
    base_url = f"{split_url.scheme}://{split_url.netloc}"
    pathway = parsed_url.path
    hostname = parsed_url.hostname
    hostname = str(hostname)
    
    # Extract subdomain, domain, and TLD using tldextract
    ext = tldextract.extract(hostname)
    subdomain = ext.subdomain
    domain = ext.domain
    tld = ext.suffix
    
    return subdomain, domain, hostname, tld, base_url, pathway, scheme


def words_raw_extraction(url):
        subdomain, domain, _, tld, _, pathway, _ = url_components(url)
        w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
        w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())   
        w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", pathway.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None,raw_words))
        return raw_words, list(filter(None,w_host)), list(filter(None,w_path))
    
    
#################################################################################################################################
#               Length Features
#################################################################################################################################

def url_length(url):
    return len(url) 

def hostname_length(hostname):
    return len(hostname) 

def count_subdomain(url):
    if len(re.findall("\.", url)) == 1:
        return 1
    elif len(re.findall("\.", url)) == 2:
        return 2
    else:
        return 3
    
#################################################################################################################################
#               Raw Word Features
#################################################################################################################################
    
def char_repeat(words_raw):
    
        def __all_same(items):
            return all(x == items[0] for x in items)

        repeat = {'2': 0, '3': 0, '4': 0, '5': 0}
        part = [2, 3, 4, 5]

        for word in words_raw:
            for char_repeat_count in part:
                for i in range(len(word) - char_repeat_count + 1):
                    sub_word = word[i:i + char_repeat_count]
                    if __all_same(sub_word):
                        repeat[str(char_repeat_count)] = repeat[str(char_repeat_count)] + 1
        return  sum(list(repeat.values()))
    
def length_word_raw(words_raw):
    return len(words_raw) 

def average_word_length(words_raw):
    if len(words_raw) ==0:
        return 0
    return sum(len(word) for word in words_raw) / len(words_raw)

def longest_word_length(words_raw):
    if len(words_raw) ==0:
        return 0
    return max(len(word) for word in words_raw) 

def shortest_word_length(words_raw):
    if len(words_raw) ==0:
        return 0
    return min(len(word) for word in words_raw)

def check_www(words_raw):
        count = 0
        for word in words_raw:
            if not word.find('www') == -1:
                count += 1
        return count

#################################################################################################################################
#               Ratio Features
#################################################################################################################################
    
def digitLetter(hostname):
    """Calculates the ratio of letters and digits in the hostname string"""
    letters = sum(c.isalpha() for c in hostname)
    digits = sum(c.isdigit() for c in hostname)
    if letters == 0:
        return 0
    digitLetterRatio = digits/letters
    return digitLetterRatio
    
def ratio_digits(hostname):
    return len(re.sub("[^0-9]", "", hostname))/len(hostname)

    
#################################################################################################################################
#               Special Characters Features
#################################################################################################################################

def count_at(base_url):
     return base_url.count('@')

def count_semicolon(url):
     return url.count(';')
 
def count_double_slash(full_url):
    list=[x.start(0) for x in re.finditer('//', full_url)]
    if list[len(list)-1]>6:
        return 1
    else:
        return 0
    return full_url.count('//')    

def count_exclamation(base_url):
    return base_url.count('?')

def count_or(url):
    return url.count('|')

def count_slash(full_url):
    return full_url.count('/')


def count_hyphens(base_url):
    return base_url.count('-')

def count_dots(hostname):
    return hostname.count('.')

def https_token(scheme):
    if scheme == 'https':
        return 0
    return 1

def abnormal_subdomain(url):
    if re.search('(http[s]?://(w[w]?|\d))([w]?(\d|-))',url):
        return 1
    return 0


#################################################################################################################################
#               Phishing Hint Features
#################################################################################################################################

suspicious_tlds = ['fit','tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu', 'online', 'click', # Spamhaus
        'country', 'stream', 'download', 'xin', 'racing', 'jetzt',
        'ren', 'mom', 'party', 'review', 'trade', 'accountants', 
        'science', 'work', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
        'accountant', 'realtor', 'top', 'christmas', 'gdn', # Shady Top-Level Domains
        'link', # Blue Coat Systems
        'asia', 'club', 'la', 'ae', 'exposed', 'pe', 'go.id', 'rs', 'k12.pa.us', 'or.kr',
        'ce.ke', 'audio', 'gob.pe', 'gov.az', 'website', 'bj', 'mx', 'media', 'sa.gov.au' # statistics
        ]


def suspicious_tld(tld):
   if tld in suspicious_tlds:
       return 1
   return 0 

HINTS = ['wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view']

def phish_hints(pathway):
    """Count the number of phishy hint in the pathway"""
    count = 0
    for hint in HINTS:
      count += pathway.lower().count(hint)
    return count

def tld_in_path(tld, path):
    if path.lower().count(tld)>0:
        return 1
    return 0

def count_http_token(url_path):
    return url_path.count('http')

#################################################################################################################################
#               Statistical Features
#################################################################################################################################

def url_entropy(url):
    freqs = Counter(url)
    probs = [float(freqs[c]) / len(url) for c in freqs]
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in probs])
    return entropy


english_chars = "abcdefghijklmnopqrstuvwxyz"
english_probs = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
                 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
                 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
                 0.00978, 0.02360, 0.00150, 0.01974, 0.00074]


def js_divergence(url):
    url_freqs = Counter(url)
    url_probs = [float(url_freqs[c]) / len(url) for c in english_chars]
    url_probs = [p + 1e-10 for p in url_probs]
    m = [(p + q) / 2.0 for p, q in zip(url_probs, english_probs)]
    js_divergence = (sum([p * math.log(p/q) for p, q in zip(url_probs, m)]) +
                     sum([q * math.log(q/p) for p, q in zip(url_probs, m)])) / 2.0
    return js_divergence
    
def ks_distance(url):
    url_freqs = Counter(url)

    url_cdf = np.cumsum([float(url_freqs[c]) / len(url) for c in english_chars])
    english_cdf = np.cumsum(english_probs)
    ks_distance = np.max(np.abs(url_cdf - english_cdf))
    return ks_distance

def euclid_distance(url):
    url_freqs = Counter(url)
    english_freqs = Counter(english_chars)

    # create a feature vector for the URL and English distributions
    url_vector = [float(url_freqs[c]) / len(url) for c in english_chars]
    english_vector = [float(english_freqs[c]) / len(english_chars) for c in english_chars]

    # calculate the Euclidean distance between the URL and English feature vectors
    euclidean_distance = np.linalg.norm(np.array(url_vector) - np.array(english_vector))
    return euclidean_distance

english = string.ascii_lowercase + string.digits + string.punctuation

def hellinger_distance(url):
    url_freq = {c: url.lower().count(c) for c in english}
    english_freq = {c: english.count(c) for c in english}
    
    url_norm = {k: v / len(url) for k, v in url_freq.items()}
    english_norm = {k: v / len(english) for k, v in english_freq.items()}
    
    bc = sum(math.sqrt(url_norm.get(k, 0) * english_norm.get(k, 0)) for k in set(english_freq) | set(url_freq))

    distance = math.sqrt(1 - bc)
    return distance

def page_rank(url):
    key = '8ccwssc0w0ogk4sc0c0wkccgs8gww4g00ogw8cgc'
    extract = tldextract.extract(url)
    domain = extract.domain
    tld = extract.suffix

    # Combine the domain and suffix into a full domain
    full_domain = domain + '.' + tld
    api_url = f'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D={full_domain}'
    headers = {'API-OPR': key}
    try:
        response = requests.get(api_url, headers=headers)
        result = response.json()['response'][0]['page_rank_integer']
        return result
    except:
        return -1
    
import tranco

# Fetch the latest Tranco list
t = tranco.Tranco(cache=True, cache_dir='.tranco')
latest_list = t.list()

def website_traffic(url):
    # Extract the domain and suffix from the URL
    extract = tldextract.extract(url)
    domain = extract.domain
    tld = extract.suffix

    # Combine the domain and suffix into a full domain
    full_domain = domain + '.' + tld

    # Get the rank of the full domain in the Tranco list
    rank = latest_list.rank(full_domain)

    # Return the rank if it exists, otherwise return -1
    if rank is not None:
        return rank
    else:
        return -1

import dns.resolver

def dns_record(url):
    # Extract the domain and suffix from the URL
    extract = tldextract.extract(url)
    domain = extract.domain
    tld = extract.suffix

    # Combine the domain and suffix into a full domain
    full_domain = domain + '.' + tld

    try:
        nameservers = dns.resolver.query(full_domain,'NS')
        if len(nameservers)>0:
            return 0
        else:
            return 1
    except:
        return 1

def extract_features(url):
    subdomain, domain, hostname, tld, base_url, pathway, scheme = url_components(url)
    raw_words, w_host, w_path = words_raw_extraction(url)
    features = {}
    """words raw features"""
    features["char_repeat"] = char_repeat(raw_words)
    features["length_word_raw"] = length_word_raw(raw_words)
    features["average_word_length"] = average_word_length(raw_words)
    features["longest_word_length"] = longest_word_length(raw_words)
    features["shortest_word_length"] = shortest_word_length(raw_words)
    features["check_www"] = check_www(raw_words)
    features["digitLetter_host"] = digitLetter(hostname)
    
    """URL Features"""
    features["count_subdomain"] = count_subdomain(url)
    features["url_length"] = url_length(url)   
    features["abnormal_subdomain"] = abnormal_subdomain(url)
    
    """Pathway features"""
    features["phish_hints"] = phish_hints(pathway)
    
    """Hostname features"""
    features["ratio_digits"] = ratio_digits(hostname)
    features["hostname_length"] = hostname_length(hostname)
    
    """Scheme features"""
    features["https_token"] = https_token(scheme)
    
    """TLD Features"""
    features["suspicious_tld"] = suspicious_tld(tld)
    features["tld_in_path"] = tld_in_path(tld, pathway)
    features["count_http_token"] = count_http_token(url)
    
    """Special Characters"""
    features["count_at"] = count_at(url)
    features["count_semicolon"] = count_semicolon(url)
    features["count_double_slash"] = count_double_slash(url)
    features["count_exclamation"] = count_exclamation(url)
    features["count_or"] = count_or(url)
    features["count_slash"] = count_slash(url)
    features["count_dots"] = count_dots(hostname)
    features["count_hyphens"] = count_hyphens(base_url)
    
    """Ratio/Count Features"""
    features["url_entropy"] = url_entropy(url)
    features["js_divergence"] = js_divergence(url)
    features["ks_distance"] = ks_distance(url)
    features["euclidean_distance"] = euclid_distance(url)
    features["hellinger_distance"] = hellinger_distance(url)
    
    
    """Rank Features"""
    features["PageRank"] = page_rank(url)
    features["search_tld"] = website_traffic(url)
    features["dns_record"] = dns_record(url)
    
    # return features
    features_array = np.array(list(features.values())).reshape(1, -1)
    
    # reshape the array to have shape (1, 27)
    features_array = np.reshape(features_array, (1, -1))
    
    features_df = pd.DataFrame(features_array, columns=list(features.keys()))
    
    return features_df













