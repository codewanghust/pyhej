import re
def name_std(text):
    """
    Name standard

    # Arguments
        text: a string

    # Returns
        A string

    # Raises
        ..
    """
    tmp = text.lower()
    tmp = re.sub(r'[()（）]', '', tmp)
    tmp = re.sub(r'[\s,，]+', ' ', tmp)
    return tmp


if __name__ == '__main__':
    print(name_std('我们  我们,我们，(我们)（我们）'))