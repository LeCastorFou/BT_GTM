from  Levenshtein import *
import re, math
from collections import Counter
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1

    res = min([LD(s[:-1], t)+1,LD(s, t[:-1])+1,LD(s[:-1], t[:-1]) + cost])
    return res

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def LDsim(s,t):
    LevenSim = 1 - LD(s,t)/(max(len(s),len(t)))
    return LevenSim

def insert_sting_middle(string, word):
	return string + word + string


WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def text_cosine(str1,str2):
    vector1 = text_to_vector(str1)
    vector2 = text_to_vector(str2)
    cosine = get_cosine(vector1, vector2)
    return cosine

def nocu(df,gb):
    df = df.groupby(gb).apply(len)
    df = pd.DataFrame(df)
    df['index'] = df.index
    df.columns = np.array(['no','index'])
    df = df.sort_values(by =['no'], ascending=False)
    return df

def sumcu(df,gb,tosum):
    df = df.groupby(gb)[[tosum]].sum()
    df = pd.DataFrame(df)
    df['index'] = df.index
    df.columns = np.array(['sum','index'])
    df = df.sort_values(by =['sum'], ascending=False)
    return df

def maxcu(df,gb,tomax):
    df = df.groupby(gb)[[tomax]].max()
    df = pd.DataFrame(df)
    df['index'] = df.index
    df.columns = np.array(['max','index'])
    df = df.sort_values(by =['max'], ascending=False)
    return df

def DiffDate(df,gb,date):
    df_max = df.groupby(gb)[[date]].max()
    df_max = pd.DataFrame(df_max)
    df_max['index'] = df_max.index
    df_max.columns = np.array(['maxdate','index'])
    df_min = df.groupby(gb)[[date]].min()
    df_min = pd.DataFrame(df_min)
    df_min['index'] = df_min.index
    df_min.columns = np.array(['mindate','index'])
    df_max['mindate'] = df_min['mindate']
    diff = df_max['maxdate'] - df_max['mindate']
    diff = list(diff)
    diff = [(w.total_seconds())/(3600*24) for w in diff]
    df_max['AgeInDays'] = diff
    df = df_max.sort_values(by =['AgeInDays'], ascending=False)
    return df
