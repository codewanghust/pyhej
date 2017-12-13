import os
import codecs


class MD(object):
    def __init__(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        if not name.endswith('.md'):
            name += '.md'

        fpath = os.path.join(path, name)
        self.writer = codecs.open(fpath, 'w', 'utf-8')

    def add_text(self, text):
        self.writer.write(text)
        self.writer.write('  \n')

    def add_code(self, code):
        self.writer.write('```\n')
        self.writer.write(code)
        self.writer.write('\n```\n')

    def add_topic(self, topic, level=2):
        self.writer.write('{} {}'.format('#'*level, topic))
        self.writer.write('\n')

    def add_image(self, fpath):
        self.writer.write('![]({})'.format(fpath))
        self.writer.write('  \n')

    def close(self):
        self.writer.close()