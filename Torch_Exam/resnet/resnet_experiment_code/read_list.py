from matplotlib import pyplot as plt
import pickle

is_plain = False
is_resnet = True
draw_plot = True

if is_plain :
    path = "/home/jsh/PycharmProjects/Torch_Exam/resnet_data/plainNet/WI+Pad"

    plain20test = []
    plain20train = []
    plain32test = []
    plain32train = []
    plain44test = []
    plain44train = []
    plain56test = []
    plain56train = []

    with open(path + '/list_plain20_testErr', 'rb') as file:
        plain20test = pickle.load(file)
    with open(path + '/list_plain20_trainErr', 'rb') as file:
        plain20train = pickle.load(file)
    with open(path + '/list_plain32_testErr', 'rb') as file:
        plain32test = pickle.load(file)
    with open(path + '/list_plain32_trainErr', 'rb') as file:
        plain32train = pickle.load(file)
    with open(path + '/list_plain44_testErr', 'rb') as file:
        plain44test = pickle.load(file)
    with open(path + '/list_plain44_trainErr', 'rb') as file:
        plain44train = pickle.load(file)
    with open(path + '/list_plain56_testErr', 'rb') as file:
        plain56test = pickle.load(file)
    with open(path + '/list_plain56_trainErr', 'rb') as file:
        plain56train = pickle.load(file)

    print("plain min error")
    print(min(plain20test))
    print(min(plain32test))
    print(min(plain44test))
    print(min(plain56test))

if is_resnet:
    path = '/home/jsh/PycharmProjects/Torch_Exam/resnet_data/resNet/weight_init+RandomCrop'
    res20test = []
    res20train = []
    res32test = []
    res32train = []
    res44test = []
    res44train = []
    res56test = []
    res56train = []

    with open(path+'/list_res20_test(weight init+padding)', 'rb') as file:
        res20test = pickle.load(file)
    with open(path+'/list_res20_train(weight init+padding)', 'rb') as file:
        res20train = pickle.load(file)
    with open(path+'/list_res32_test(weight init+padding)', 'rb') as file:
        res32test = pickle.load(file)
    with open(path+'/list_res32_train(weight init+padding)', 'rb') as file:
        res32train = pickle.load(file)
    with open(path+'/list_res44_test(weight init+padding)', 'rb') as file:
        res44test = pickle.load(file)
    with open(path+'/list_res44_train(weight init+padding)', 'rb') as file:
        res44train = pickle.load(file)
    with open(path+'/list_res56_test(weight init+padding)', 'rb') as file:
        res56test = pickle.load(file)
    with open(path+'/list_res56_train(weight init+padding)', 'rb') as file:
        res56train = pickle.load(file)

    print("resnet min error")
    print(min(res20test))
    print(min(res32test))
    print(min(res44test))
    print(min(res56test))

if draw_plot:
    # draw plot
    if is_plain:
        line20test = plt.plot(plain20test)
        line20train = plt.plot(plain20train)
        line32test = plt.plot(plain32test)
        line32train = plt.plot(plain32train)
        line44test = plt.plot(plain44test)
        line44train = plt.plot(plain44train)
        line56test = plt.plot(plain56test)
        line56train = plt.plot(plain56train)

        plt.setp(line20test, color='r', linewidth=1.5)
        plt.setp(line20train, color='r', linewidth=0.5)
        plt.setp(line32test, color='b', linewidth=1.5)
        plt.setp(line32train, color='b', linewidth=0.5)
        plt.setp(line44test, color='g', linewidth=1.5)
        plt.setp(line44train, color='g', linewidth=0.5)
        plt.setp(line56test, color='k', linewidth=1.5)
        plt.setp(line56train, color='k', linewidth=0.5)

        plt.ylabel('Error')
        plt.ylim((0, 0.4))
        plt.xlabel('epoch')
        plt.legend(['plain20test','plain20train','plain32test','plain32train',
                    'plain44test','plain44train','plain56test','plain56train'])
        plt.title('train, test error(plainNet)')
        plt.savefig(path + '/train, test error(plainNet)_WI+padding.png')
        plt.show()

    if is_resnet:
        line_res20test = plt.plot(res20test)
        line_res20train = plt.plot(res20train)
        line_res32test = plt.plot(res32test)
        line_res32train = plt.plot(res32train)
        line_res44test = plt.plot(res44test)
        line_res44train = plt.plot(res44train)
        line_res56test = plt.plot(res56test)
        line_res56train = plt.plot(res56train)

        plt.setp(line_res20test,color='r',linewidth=1.5)
        plt.setp(line_res20train,color='r',linewidth=0.5)
        plt.setp(line_res32test,color='b',linewidth=1.5)
        plt.setp(line_res32train,color='b',linewidth=0.5)
        plt.setp(line_res44test,color='g',linewidth=1.5)
        plt.setp(line_res44train,color='g',linewidth=0.5)
        plt.setp(line_res56test,color='k',linewidth=1.5)
        plt.setp(line_res56train,color='k',linewidth=0.5)

        plt.ylabel('Error')
        plt.ylim((0.0,0.5))
        plt.xlabel('epoch')
        plt.legend(['res20test','res20train','res32test','res32train',
                    'res44test','res44train','res56test','res56train'])
        plt.title('train, test error(resNet)')
        plt.savefig(path+'/train, test error(resNet)_padding(0.1~0.13).png')
        plt.show()

