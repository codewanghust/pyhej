import os
import re
import shutil
import requests
from six.moves.urllib.request import urlretrieve


URL_REGEX = re.compile(r'http://|https://|ftp://')


def read(filename, default=None):
    try:
        if URL_REGEX.match(filename) is not None:
            return requests.get(filename).content
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def download(furls, outdir='tmps', prefix=''):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    for i, furl in enumerate(furls):
        fpath = os.path.join(outdir, '{}{08d}.jpg'.format(prefix, i))
        try:
            urlretrieve(furl, fpath)
        except Exception:
            print('* Fail | [{}, {}]'.format(i, furl))

