import os
import shutil

dir_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/img_align_celeba/'
count = 0
test_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/test/celeb/'
train_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/train/celeb/'

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        ruta = os.path.join(dir_path, path)
        new_name = 'celebpy' + str(count) + '.jpg'
        file_name = os.path.basename(ruta)
        if count < 128:
            file_name = shutil.copy(ruta, train_path + new_name)
        else:
            file_name = shutil.copy(ruta, test_path + new_name)
    count += 1
    if count == 160:
        break
    print(count)