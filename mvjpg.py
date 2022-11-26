import os
dir_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/Fotos_Alan/'
count = 0
test_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/test/'
train_path = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/train/'
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        ruta = os.path.join(dir_path, path)
        print(ruta)
        new_name = 'py' + str(count) + '.jpg'
        file_name = os.path.basename(ruta)
        if count < 800:
            file_name = os.rename(ruta, train_path + new_name)

        else:
            file_name = os.rename(ruta, test_path + new_name)
    count += 1
print('File count:', count)
