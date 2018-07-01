import os
import time
def generateFileList(root_dir):

    print("Generating the list file, train.txt and validation.txt")

    groups = os.listdir(root_dir)
    fid_train = open("train_update.txt", 'w')
    fid_test = open("test_update.txt", 'w')

    for subset, fid in zip(['train', 'test'], [fid_train, fid_test]):
        subsetpath = os.path.join(root_dir, subset)
        cls = os.listdir(os.path.join(subsetpath))
        cls = [tmp for tmp in cls if 'Store' not in tmp]
        cls = sorted(cls)
        print(cls)
        # time.sleep(10)
        for i,class_name in enumerate(cls):
            basedir = os.path.join(subsetpath, class_name)
            filename = os.listdir(basedir)
            for tmpfile in filename:
                if 'jpeg' not in tmpfile:
                    continue
                fid.write(os.path.join(basedir, tmpfile) + ' ' + str(i) + '\n')

    fid_train.close()
    fid_test.close()


if __name__ == '__main__':
    # root_dir = '/mnt/Medical/DR/resize'
    root_dir = '/mnt/Medical/DR/ben'
    generateFileList(root_dir)
