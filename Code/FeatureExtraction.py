# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:47:48 2023

@author: Daniel
"""

from urllib.parse import urlparse, urlsplit
import tldextract
import math
from math import log2
import re
from spellchecker import SpellChecker

def url_components(url):
    """Returns the subdomain, domain, hostname, TLD, base URL, pathway, and scheme of a URL."""
    parsed_url = urlparse(url)
    split_url = urlsplit(url)
    base_url = f"{split_url.scheme}://{split_url.netloc}"
    pathway = parsed_url.path
    hostname = parsed_url.hostname
    
    # Extract subdomain, domain, and TLD using tldextract
    ext = tldextract.extract(hostname)
    subdomain = ext.subdomain
    domain = ext.domain
    tld = ext.suffix
    
    return subdomain, domain, hostname, tld, base_url, pathway

def entropy(hostname):
    """Calculates the Shannon entropy of a string in subdomain"""
    prob = [ float(hostname.count(c)) / len(hostname) for c in dict.fromkeys(list(hostname)) ]
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
    return entropy

def dom_entropy(domain):
    """Calculates the Shannon entropy of a string in subdomain"""
    prob = [ float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain)) ]
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
    return entropy

def sub_entropy(subdomain):
    """Calculates the Shannon entropy of a string in subdomain"""
    prob = [ float(subdomain.count(c)) / len(subdomain) for c in dict.fromkeys(list(subdomain)) ]
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
    return entropy

# Reference English language character distribution
english_distribution = {'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253, 'e': 0.12702, 'f': 0.02228, 'g': 0.02015, 'h': 0.06094, 'i': 0.06094, 'j': 0.00153, 'k': 0.00772, 'l': 0.04025, 'm': 0.02406, 'n': 0.06749, 'o': 0.07507, 'p': 0.01929, 'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056, 'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150, 'y': 0.01974, 'z': 0.00074}

def kl_divergence(hostname):
    """Calculates the Kullback–Leibler divergence in the hostname string"""
    hostname_distribution = {}
    for char in english_distribution.keys():
        if len(hostname) == 0:
            return 0
        hostname_distribution[char] = hostname.count(char) / len(hostname)
    kl_div = 0
    for char in english_distribution.keys():
        if hostname_distribution.get(char) != None and hostname_distribution[char] != 0:
            kl_div +=  hostname_distribution[char] * log2( hostname_distribution[char] / english_distribution[char] )
    return kl_div

def dom_kl_divergence(domain):
    """Calculates the Kullback–Leibler divergence in the domain string"""
    domain_distribution = {}
    for char in english_distribution.keys():
        if len(domain) == 0:
            return 0
        domain_distribution[char] = domain.count(char) / len(domain)
    kl_div = 0
    for char in english_distribution.keys():
        if domain_distribution.get(char) != None and domain_distribution[char] != 0:
            kl_div +=  domain_distribution[char] * log2( domain_distribution[char] / english_distribution[char] )
    return kl_div

def sub_kl_divergence(subdomain):
    """Calculates the Kullback–Leibler divergence in the subdomain string"""
    sub_domain_distribution = {}
    for char in english_distribution.keys():
        if len(subdomain) == 0:
            return 0
        sub_domain_distribution[char] = subdomain.count(char) / len(subdomain)
    kl_div = 0
    for char in english_distribution.keys():
        if sub_domain_distribution.get(char) != None and sub_domain_distribution[char] != 0:
            kl_div +=  sub_domain_distribution[char] * log2( sub_domain_distribution[char] / english_distribution[char] )
    return kl_div

def digit_ratio(hostname):
    """Calculates the ratio of digits in the hostname string"""
    if not re.search('[^0-9]', hostname):
        return 0
    return len(re.sub("[^0-9]", "", hostname))/len(hostname)

def dom_digit_ratio(domain):
    """Calculates the ratio of digits in the hostname string"""
    if not re.search('[^0-9]', domain):
        return 0
    return len(re.sub("[^0-9]", "", domain))/len(domain)

def sub_digit_ratio(subdomain):
    if not re.search('[^0-9]', subdomain):
        return 0
    """Calculates the ratio of digits in the hostname string"""
    return len(re.sub("[^0-9]", "", subdomain))/len(subdomain)

def letter_ratio(hostname):
    if not re.search('[a-z]', hostname):
        return 0
    """Calculates the ratio of letters in the hostname string"""
    return len(re.sub("[a-z]", "", hostname))/len(hostname)

def dom_letter_ratio(domain):
    if not re.search('[a-z]', domain):
        return 0
    """Calculates the ratio of letters in the hostname string"""
    return len(re.sub("[a-z]", "", domain))/len(domain)

def sub_letter_ratio(subdomain):
    if not re.search('[a-z]', subdomain):
        return 0
    """Calculates the ratio of letters in the hostname string"""
    return len(re.sub("[a-z]", "", subdomain))/len(subdomain)

def digitLetter(hostname):
    """Calculates the ratio of letters and digits in the hostname string"""
    letters = sum(c.isalpha() for c in hostname)
    digits = sum(c.isdigit() for c in hostname)
    if letters == 0:
        return 0
    digitLetterRatio = digits/letters
    return digitLetterRatio

def dom_digitLetter(domain):
    """Calculates the ratio of letters and digits in the hostname string"""
    letters = sum(c.isalpha() for c in domain)
    digits = sum(c.isdigit() for c in domain)
    if letters == 0:
        return 0
    digitLetterRatio = digits/letters
    return digitLetterRatio

def sub_digitLetter(subdomain):
    """Calculates the ratio of letters and digits in the hostname string"""
    letters = sum(c.isalpha() for c in subdomain)
    digits = sum(c.isdigit() for c in subdomain)
    if letters == 0:
        return 0
    digitLetterRatio = digits/letters
    return digitLetterRatio

def vowel_con_ratio(hostname):
    vowels = sum(1 for char in hostname if char.casefold() in 'aeiou')
    consonants = sum(1 for char in hostname if char.casefold() in 'qwrtypsdfghjklzxcvbnm')
    if consonants == 0:
        return 1.0  # 100% ratio of vowels if the hostname only contains letters
    else:
        vow_con_ratio = vowels / consonants
        return vow_con_ratio
    
def dom_vowel_con_ratio(domain):
    vowels = sum(1 for char in domain if char.casefold() in 'aeiou')
    consonants = sum(1 for char in domain if char.casefold() in 'qwrtypsdfghjklzxcvbnm')
    if consonants == 0:
        return 1.0  # 100% ratio of vowels if the hostname only contains letters
    else:
        vow_con_ratio = vowels / consonants
        return vow_con_ratio
    
def sub_vowel_con_ratio(subdomain):
    vowels = sum(1 for char in subdomain if char.casefold() in 'aeiou')
    consonants = sum(1 for char in subdomain if char.casefold() in 'qwrtypsdfghjklzxcvbnm')
    if consonants == 0:
        return 1.0  # 100% ratio of vowels if the hostname only contains letters
    else:
        vow_con_ratio = vowels / consonants
        return vow_con_ratio

def digits_presence(hostname):
  """Check if the numbers are present in subdomains"""
  if re.findall("\d", hostname):
    return 1
  return 0

def dom_digits_presence(domain):
  """Check if the numbers are present in subdomains"""
  if re.findall("\d", domain):
    return 1
  return 0

def sub_digits_presence(subdomain):
  """Check if the numbers are present in subdomains"""
  if re.findall("\d", subdomain):
    return 1
  return 0

def shortest_word_length(hostname):
  """Check if the length of shortest word in hostname"""
  if len(re.findall("\w+", hostname)) == 0:
      return 0
  return min(len(word) for word in re.findall("\w+", hostname)) 

def longest_word_length(hostname):
    """Check if the length of longest word in hostname"""
    if len(re.findall("\w+", hostname)) ==0:
        return 0
    return max(len(word) for word in re.findall("\w+", hostname)) 

def average_word_length(hostname):
    """Calculates the average word length in the hostname"""
    if len(re.findall("\w+", hostname)) ==0:
        return 0
    return sum(len(word) for word in re.findall("\w+", hostname)) / len(re.findall("\w+", hostname))

def char_repeat(hostname):
    """Count repeated characters"""
    repeat_counts = [0] * 4  # [2, 3, 4, 5]
    
    for word in re.findall("\w+", hostname):
        for char_repeat_count in range(2, 6):
            for i in range(len(word) - char_repeat_count + 1):
                sub_word = word[i:i + char_repeat_count]
                if len(set(sub_word)) == 1:
                    repeat_counts[char_repeat_count-2] += 1
    return sum(repeat_counts)

def dom_char_repeat(domain):
    """Count repeated characters"""
    repeat_counts = [0] * 4  # [2, 3, 4, 5]
    
    for word in re.findall("\w+", domain):
        for char_repeat_count in range(2, 6):
            for i in range(len(word) - char_repeat_count + 1):
                sub_word = word[i:i + char_repeat_count]
                if len(set(sub_word)) == 1:
                    repeat_counts[char_repeat_count-2] += 1
    return sum(repeat_counts)

def sub_char_repeat(subdomain):
    """Count repeated characters"""
    repeat_counts = [0] * 4  # [2, 3, 4, 5]
    
    for word in re.findall("\w+", subdomain):
        for char_repeat_count in range(2, 6):
            for i in range(len(word) - char_repeat_count + 1):
                sub_word = word[i:i + char_repeat_count]
                if len(set(sub_word)) == 1:
                    repeat_counts[char_repeat_count-2] += 1
    return sum(repeat_counts)

def hostname_length(hostname):
    """Count the length of hostname"""
    return len(hostname) 

def url_length(url):
    """Count the length of url"""
    return len(url) 

HINTS = ['wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view']

def phish_hints(pathway):
    """Count the number of phishy hint in the pathway"""
    count = 0
    for hint in HINTS:
      count += pathway.lower().count(hint)
    return count

spell = SpellChecker(language='en')

def word_in_dict(pathway):
    if not re.search('[a-zA-Z]', pathway):
        return 0
    pathway = re.sub(r'[^a-zA-Z]', ' ', pathway)
    pathway = pathway.replace(" ", "")
    words = {word.lower() for i in range(len(pathway)) for j in range(i+1, len(pathway)+1) 
              for word in [pathway[i:j]] if word.lower() in spell}
    return len(words)

def hostname_dictionary_words(hostname):
    if not re.search('[a-zA-Z]', hostname):
        return 0
    hostname = re.sub(r'[^a-zA-Z]', ' ', hostname)
    hostname = hostname.replace(" ", "")
    words = {word.lower() for i in range(len(hostname)) for j in range(i+1, len(hostname)+1) 
             for word in [hostname[i:j]] if word.lower() in spell}
    return len(words)

def count_digits(hostname, pathway):
  return len(re.sub("[^0-9]", "", hostname + pathway))

def count_hyphens(base_url):
    return base_url.count('-')

def count_subdomain(url):
    if len(re.findall("\.", url)) == 1:
        return 1
    elif len(re.findall("\.", url)) == 2:
        return 2
    else:
        return 3

def count_dots(base_url):
  return base_url.count('.')

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

def tld_in_path(tld, pathway):
    if pathway.lower().count(tld)>0:
        return 1
    return 0

def tld_in_subdomain(tld, subdomain):
    if subdomain.count(tld)>0:
        return 1
    return 0

def tld_in_bad_position(tld, subdomain, pathway):
    if tld_in_path(tld, pathway)== 1 or tld_in_subdomain(tld, subdomain)==1:
        return 1
    return 0

# from word_with_nlp import nlp_class

# def random_domain(domain):
#         nlp_manager = nlp_class()
#         return nlp_manager.check_word_random(domain)

def check_www(url):
    count = 0
    for word in re.findall("\w+", url):
        if not word.find('www') == -1:
            count += 1
    return count

def check_com(url):
        count = 0
        for word in re.findall("\w+", url):
            if not word.find('com') == -1:
                count += 1
        return count


def port(url):
    if re.search("^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",url):
        return 1
    return 0

def prefix_suffix(url):
    if re.findall(r"https?://[^\-]+-[^\-]+/", url):
        return 1
    else:
        return 0 
    
suspecious_tlds = ['fit','tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu', 'online', 'click', # Spamhaus
        'country', 'stream', 'download', 'xin', 'racing', 'jetzt',
        'ren', 'mom', 'party', 'review', 'trade', 'accountants', 
        'science', 'work', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
        'accountant', 'realtor', 'top', 'christmas', 'gdn', # Shady Top-Level Domains
        'link', # Blue Coat Systems
        'asia', 'club', 'la', 'ae', 'exposed', 'pe', 'go.id', 'rs', 'k12.pa.us', 'or.kr',
        'ce.ke', 'audio', 'gob.pe', 'gov.az', 'website', 'bj', 'mx', 'media', 'sa.gov.au' # statistics
        ]


def suspecious_tld(tld):
   if tld in suspecious_tlds:
       return 1
   return 0

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
    
    
def abnormal_subdomain(url):
    if re.search('(http[s]?://(w[w]?|\d))([w]?(\d|-))',url):
        return 1
    return 0

def extract_features(urls):
    features_list = []
    for url in urls:
        subdomain, domain, hostname, tld, base_url, pathway = url_components(url)
        features = []
        features.append(entropy(hostname))
        features.append(dom_entropy(domain))
        features.append(sub_entropy(subdomain))
        features.append(kl_divergence(hostname))
        features.append(dom_kl_divergence(domain))
        features.append(sub_kl_divergence(subdomain))
        features.append(digit_ratio(hostname))
        features.append(dom_digit_ratio(domain))
        features.append(sub_digit_ratio(subdomain))
        features.append(letter_ratio(hostname))
        features.append(dom_letter_ratio(domain))
        features.append(sub_letter_ratio(subdomain))
        features.append(digitLetter(hostname))
        features.append(dom_digitLetter(domain))
        features.append(sub_digitLetter(subdomain))
        features.append(vowel_con_ratio(hostname))
        features.append(dom_vowel_con_ratio(domain))
        features.append(sub_vowel_con_ratio(subdomain))
        features.append(digits_presence(subdomain))
        features.append(shortest_word_length(hostname))
        features.append(longest_word_length(hostname))
        features.append(average_word_length(hostname))
        features.append(char_repeat(hostname))
        features.append(dom_char_repeat(domain))
        features.append(sub_char_repeat(subdomain))
        features.append(hostname_length(hostname))
        features.append(url_length(hostname))
        features.append(phish_hints(pathway))
        features.append(word_in_dict(pathway))
        features.append(hostname_dictionary_words(hostname))
        features.append(count_digits(hostname, pathway))
        features.append(count_hyphens(base_url))
        features.append(count_subdomain(url))
        features.append(count_dots(base_url))
        features.append(shortening_service(url))
        features.append(tld_in_path(tld, pathway))
        features.append(tld_in_subdomain(tld, subdomain))
        features.append(tld_in_bad_position(tld, subdomain, pathway))
        features.append(check_www(url))
        features.append(check_com(url))
        features.append(prefix_suffix(url))
        features.append(suspecious_tld(tld))
        features.append(having_ip_address(url))
        features.append(abnormal_subdomain(url))
        features_list.append(features)
        
    return features_list

# url = input("URL: ")
# user_input = extract_features(url)