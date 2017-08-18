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

