# 导入需要的包
import random
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count
import numpy as np
from PIL import Image


class MyReader:
    def __init__(self, imageSize):
        self.imageSize = imageSize

    def load_image(self, file):
        # 打开图片
        img = Image.open(file)
        img = img.resize((self.imageSize, self.imageSize), Image.ANTIALIAS)
        # 随机水平翻转
        r1 = random.random()
        if r1 > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 随机垂直翻转
        r2 = random.random()
        if r2 > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 随机角度翻转
        r3 = random.randint(-3, 3)
        img = img.rotate(r3, expand=False)
        # 随机裁剪
        r4 = random.randint(0, int(self.imageSize*1.1 - self.imageSize))
        r5 = random.randint(0, int(self.imageSize*1.1 - self.imageSize))
        box = (r4, r5, r4 + self.imageSize, r5 + self.imageSize)
        img = img.crop(box)
        im = np.array(img).astype(np.float32)
        # 矩阵转置
        im = im.transpose((2, 0, 1))
        # 将像素值从【0-255】转换为【0-1】
        im = im / 255.0
        # print(im)
        im = np.expand_dims(im, axis=0)
        return im

    def train_mapper(self, sample):
        img_path, label = sample
        img_path = self.load_image(img_path)
        return img_path.flatten(), label

    # 获取训练的reader
    def train_reader(self, train_list_path, buffered_size=1024):
        def reader():
            with open(train_list_path, 'r') as f:
                # lines = [line.split() for line in f]
                lines = f.readlines()
                np.random.shuffle(lines)
                for line in lines:
                    img_path, label = line.split('\t')
                    yield img_path, int(label)

        return paddle.reader.xmap_readers(self.train_mapper, reader, cpu_count(), buffered_size)

    # 测试图片的预处理
    def test_mapper(self, sample):
        img, label = sample
        img = self.load_image(img)
        return img.flatten(), label

    # 测试的图片reader
    def test_reader(self, test_list_path, buffered_size=1024):
        def reader():
            with open(test_list_path, 'r') as f:
                # lines = [line.split() for line in f]
                lines = f.readlines()
                np.random.shuffle(lines)
                for line in lines:
                    img_path, label = line.split('\t')
                    yield img_path, int(label)

        return paddle.reader.xmap_readers(self.test_mapper, reader, cpu_count(), buffered_size)


BATCH_SIZE = 50
imgSize = 100
myreader = MyReader(imgSize)
# 用于训练的数据提供器
train_reader = paddle.batch(
    reader=myreader.train_reader('E:/Data/G_num/Dataset/train.list'), batch_size=BATCH_SIZE)
# 用于测试的数据提供器
test_reader = paddle.batch(
    reader=myreader.test_reader('E:/Data/G_num/Dataset/test.list'), batch_size=BATCH_SIZE)

# 定义输入数据
data_shape = [3, imgSize, imgSize]
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


def convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=7,  # 滤波器的大小
        num_filters=20,
        pool_size=3,  # 池化核大小3*3
        pool_stride=3,  # 池化步长
        act="relu")  # 激活类型
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=7,
        num_filters=50,
        pool_size=3,
        pool_stride=3,
        act="relu")
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    # 第三个卷积-池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=7,
        num_filters=50,
        pool_size=3,
        pool_stride=3,
        act="relu")
    # 创建一层全连接层
    hidden1 = fluid.layers.fc(input=conv_pool_3, size=50, act='relu')
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=hidden1, size=10, act='softmax')
    return prediction


# 获取分类器，用cnn进行分类
predict = convolutional_neural_network(images)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)  # 交叉熵
avg_cost = fluid.layers.mean(cost)  # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)  # 使用输入和标签计算准确率

# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
print("完成")

# 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 获取测试程序
# test_program = fluid.default_main_program().clone(for_test=True)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(feed_list=[images, label], place=place)

all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


EPOCH_NUM = 20
model_save_dir = "E:/Python Project/Gesture_cnn"

for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 喂入一个batch的数据
                                        fetch_list=[avg_cost, acc])  # fetch均方误差和准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每100次batch打印一次训练、进行一次测试
        if batch_id % 10 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 开始测试
    test_costs = []  # 测试的损失值
    test_accs = []  # 测试的准确率
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_costs.append(test_cost[0])  # 记录每个batch的误差
        test_accs.append(test_acc[0])  # 记录每个batch的准确率

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 计算误差平均值（误差和/误差的个数）
    test_acc = (sum(test_accs) / len(test_accs))  # 计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['images'],
                              [predict],
                              exe)
print('训练模型保存完成！')
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")



