import matplotlib.pyplot as plt
from pyhej.image import load_img, img_to_array
def get_plt_show(fpaths, col=5, row=None, target_size=None):
    if row is None:
        row = int(len(fpaths)/col) + 1
    plt.figure(figsize=(18, int(row/col*18)))
    for i in range(row):
        for j in range(col):
            num = i*col + j
            if len(fpaths) > num:
                ax = plt.subplot(row, col, num + 1)
                img = load_img(fpaths[num], target_size=target_size)
                arr = img_to_array(img)
                plt.imshow(arr)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else:
                break
    return plt

