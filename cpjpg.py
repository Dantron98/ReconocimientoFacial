import os
import shutil

dir_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/data/fotos_Alan/'
count = 0
test_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/test/yo/'
train_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/train/yo/'

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        ruta = os.path.join(dir_path, path)
        new_name = 'yo1py' + str(count) + '.jpg'
        file_name = os.path.basename(ruta)
        if count < 800:
            file_name = shutil.copy(ruta, train_path + new_name)
        else:
            file_name = shutil.copy(ruta, test_path + new_name)
    count += 1
    if count == 1000:
        break
    print(count)