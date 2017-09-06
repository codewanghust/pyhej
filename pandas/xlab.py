import json
import pandas as pd
from ..nlp.string import normalize


def df_json_loads(df, column, names):
    cols = []
    for text in df[column]:
        try:
            maps = json.loads(text)
            cols.append([maps.get(name, '') for name in names])
        except Exception as err:
            cols.append(['' for name in names])
    return pd.DataFrame(cols, columns=names)


def df_str_normalize(df, columns):
    for column in columns:
        df[column] = df[column].apply(normalize)
    return df


def groupx(words):
    maps = {}
    for word in words:
        maps[word] = maps.get(word, 0) + 1
    temp = sorted(maps.items(), key=lambda x: x[1], reverse=True)
    return (temp[0][0], temp[0][1], sum(maps.values()), maps)


def mergex(groups, threshold=0.8):
    # [[idx,maps,subs,bool],..]
    pairs = []
    total = len(groups)
    for i in range(total):
        igroup = groups[i]
        if igroup[-1]:
            for j in range(i+1, total):
                jgroup = groups[j]
                if jgroup[-1]:
                    nkeys = igroup[1].keys() & jgroup[1].keys()
                    s = len(nkeys) / min(len(igroup[1]), len(jgroup[1]))
                    if s > threshold:
                        pairs.append((i, j, s))
    if len(pairs) > 0:
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        for i, j, s in pairs:
            igroup = groups[i]
            jgroup = groups[j]
            if igroup[-1] and jgroup[-1]:
                igroup[-1] = False
                jgroup[-1] = False
                umaps = {}
                for key in igroup[1].keys() | jgroup[1].keys():
                    umaps[key] = igroup[1].get(key, 0) + jgroup[1].get(key, 0)
                groups.append([len(groups), umaps, igroup[2] + ',' + jgroup[2], True])
    return len(groups) - total


def mergex_weight(groups, threshold=0.8):
    # [[idx,maps,subs,bool],..]
    pairs = []
    total = len(groups)
    for i in range(total):
        igroup = groups[i]
        if igroup[-1]:
            for j in range(i+1, total):
                jgroup = groups[j]
                if jgroup[-1]:
                    nmaps = {}
                    for key in igroup[1].keys() & jgroup[1].keys():
                        nmaps[key] = min(igroup[1].get(key, 0), jgroup[1].get(key, 0))
                    s = sum(nmaps.values()) / min(sum(igroup[1].values()), sum(jgroup[1].values()))
                    if s > threshold:
                        pairs.append((i, j, s))
    if len(pairs) > 0:
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        for i, j, s in pairs:
            igroup = groups[i]
            jgroup = groups[j]
            if igroup[-1] and jgroup[-1]:
                igroup[-1] = False
                jgroup[-1] = False
                umaps = {}
                for key in igroup[1].keys() | jgroup[1].keys():
                    umaps[key] = igroup[1].get(key, 0) + jgroup[1].get(key, 0)
                groups.append([len(groups), umaps, igroup[2] + ',' + jgroup[2], True])
    return len(groups) - total

