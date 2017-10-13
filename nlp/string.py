import re
import itertools


def full2half(word):
    chars = []
    for c in word:
        num = ord(c)
        if num == 0x3000:
            chars.append(chr(0x20))
        elif 0xFF01 <= num <= 0xFF5E:
            chars.append(chr(num - 0xFEE0))
        else:
            chars.append(c)
    return ''.join(chars)


def normalize(text):
    if isinstance(text, str):
        tmp = full2half(text)
        tmp = tmp.lower()
        tmp = re.sub(r'[\s,]+', ' ', tmp)
        return tmp.strip()
    else:
        return ''


class Converter(object):
    '''全角转半角
    半角字符:unicode编码从33到126,十六进制0x21到0x7E
    全角字符:unicode编码从65281到65374,十六进制0xFF01到0xFF5E
    特殊字符:空格,全角为12288(0x3000),半角为32(0x20)
    除空格外,关系有`半角 + 65248 = 全角`
    '''
    def __init__(self):
        pass

    def full2half(self, word):
        chars = []
        for c in word:
            num = ord(c)
            if num == 0x3000:
                chars.append(chr(0x20))
            elif 0xFF01 <= num <= 0xFF5E:
                chars.append(chr(num - 0xFEE0))
            else:
                chars.append(c)
        return ''.join(chars)

    def full2halfs(self, words):
        return [self.full2half(word) for word in words]

    def half2full(self, word):
        chars = []
        for c in word:
            num = ord(c)
            if num == 0x20:
                chars.append(chr(0x3000))
            elif 0x21 <= num <= 0x7E:
                chars.append(chr(num + 0xFEE0))
            else:
                chars.append(c)
        return ''.join(chars)

    def half2fulls(self, word):
        return [self.half2full(word) for word in words]


class Counter(object):
    '''字符统计器
    排除字符,可选:汉字,字母,数字
    '''
    def __init__(self):
        self.data = {}

    def add(self, word):
        for c in word:
            self.data[c] = 1 + self.data.get(c, 0)

    def adds(self, words):
        for word in words:
            self.add(word)

    def reset(self):
        self.data = {}

    def to_list(self, reverse=False):
        temp = []
        for key, val in self.data.items():
            temp.append((key, val))
        temp = sorted(temp, key=lambda x: x[1], reverse=False)
        return temp

    def to_list_sel(self, han=False, eng=False, num=False, reverse=False):
        temp = []
        for key, val in self.data.items():
            if 48 <= ord(key) <= 57:
                if num:
                    temp.append((key, val))
            elif 65 <= ord(key) <= 90:
                if eng:
                    temp.append((key, val))
            elif 97 <= ord(key) <= 122:
                if eng:
                    temp.append((key, val))
            elif 19968 <= ord(key) <= 40869:
                if han:
                    temp.append((key, val))
            else:
                temp.append((key, val))
        temp = sorted(temp, key=lambda x: x[1], reverse=False)
        return temp


def tool_char_tagging(c):
    i = ord(c)
    if 48 <= i <= 57:
        return 'd'
    elif 65 <= i <= 90:
        return 'en'
    elif 97 <= i <= 122:
        return 'en'
    elif 19968 <= i <= 40869:
        return 'zh'
    else:
        return 'o'


def tool_text_group(text):
    res = []
    tmp, tag = [], ''
    for c in text:
        tag_cur = tool_char_tagging(c)
        if tag_cur == tag:
            tmp.append(c)
        else:
            res.append((tag, ''.join(tmp)))
            tmp, tag = [c], tag_cur
    res.append((tag, ''.join(tmp)))
    return res


re_none = re.compile('\S', re.U)
re_split = re.compile('([0-9]+|[a-z-]+|[\u2E80-\u9FFF]+)', re.U)
def tool_text_split(text):
    return [i for i in re_split.split(text) if re_none.match(i)]


re_sub_r1 = re.compile('\s*(?:\d+)\s*(?:a|ma|g|kg|t|m|mm|cm|dm|km)(?:[^a-z]|$)', re.U)
re_sub_r2 = re.compile('(^|[^a-z0-9*])(?:i|ii|iii|iv|v|vi|vii|viii)([^a-z0-9*]|$)', re.U)
def tool_text_sub(text):
    text = re_sub_r1.sub(r'*', text)
    text = re_sub_r2.sub(r'\1*\2', text)
    return text

