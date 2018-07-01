def duplicateSamples(sampleList, targetNum=700):

    sampleNum = len(sampleList)

    scale, remain = targetNum // sampleNum, targetNum % sampleNum

    # if sample number is greater than target number, return # targetNum
    if scale  == 0:
        return sampleList[:targetNum]

    if remain == 0:
        return sampleList * scale
    else:
        return sampleList * scale + sampleList[:remain]

def generateList(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # init dictionary for saving all the images.
    cls_dict = {}
    for line in lines:
        splt = line.split(' ')
        path, label = splt[0], int(splt[1])

        if label not in cls_dict.keys():
            cls_dict[label] = [path]
        else:
            cls_dict[label].append(path)


    return cls_dict     # save all the data by class

def generateData(cls_dict, step=None, dpFlag=False, targetNum=700):
    # shuffle the list and pick 1/20 samples
    pathset = []
    labelset = []
    
    random.seed(2222)

    for key in cls_dict.keys():
        sample_num = len(cls_dict[key])
        #print("The {}-th class has {:5d} samples before downsample.".format(key, sample_num))
        #print("shuffle the list and pick 1/20 samples")
        random.shuffle(cls_dict[key])
        if step is not None:
            cls_dict[key] = cls_dict[key][::step]

        if dpFlag:
            cls_dict[key] = duplicateSamples(sampleList=cls_dict[key], targetNum=targetNum)


        sample_num = len(cls_dict[key])
        #print("The {:5}-th class has {:5d} samples after downsample.".format(key, sample_num))
        #print("First 3 samples\n {}".format(cls_dict[key][:3]))

        # get the downsampled list 
        pathset += cls_dict[key]
        labelset += sample_num * [key]


    pathAndLabel = zip(pathset, labelset)
    random.shuffle(pathAndLabel)


    pathset = []
    labelset = []

    for tmp in pathAndLabel:
        pathset.append(tmp[0])
        labelset.append(tmp[1])


    return pathset, labelset