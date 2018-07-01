# counter data]
def generateList(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # init dictionary for saving all the images.
    counter = [0, 0, 0, 0, 0]
    cls_dict = {}
    for line in lines:
        splt = line.split(' ')
        path, label = splt[0], int(splt[1])
        counter[label] += 1

        if label not in cls_dict.keys():
            cls_dict[label] = [path]
        else:
            cls_dict[label].append(path)

    print(counter)

    return cls_dict     # save all the data by class


filename = 'test.txt'

generateList(filename)