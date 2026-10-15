# 导入os模块：用于操作系统文件、文件夹路径，遍历目录，删除文件
import os
# 从PIL库导入Image类：PIL(Pillow)是python图像处理库，用来打开、校验图片
from PIL import Image
# 从torchvision库导入transforms：PyTorch图像变换/数据增强工具包
from torchvision import transforms
from torchvision.datasets import ImageFolder # 按文件夹分类读取图片数据集
from torch.utils.data import DataLoader   # 构建批次加载器
from torch.nn import Conv2d
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, Linear
import matplotlib.pyplot as plt

# ===================== 配置区域，修改这里 =====================
# 数据集根目录，对应 flower 文件夹
DATA_ROOT = "./flower"
# 需要检查的子文件夹：训练集train、测试集test
SUB_FOLDERS = ["train", "test"]
# 需要校验的图片后缀，支持jpg/jpeg（小写匹配，代码里会统一转小写判断）
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
# DRY_RUN=True：只打印发现的损坏图片，【不会真正删除】；
# DRY_RUN=False：扫描到坏图直接删除！⚠️删除不可恢复！
DRY_RUN = False    #找到图片直接删除
# ============================================================

def scan_broken_images(folder_path):
    """
    递归检查目录下的图片。
    检查 verify() 和 load() 是否能够正常执行。
    DRY_RUN=False 时删除无法正常读取的图片。
    """

    broken_count = 0
    total_check = 0
    deleted_count = 0

    if not os.path.isdir(folder_path):
        print(f"[警告] 目录不存在：{folder_path}")
        return broken_count, total_check, deleted_count

    for root, dirs, files in os.walk(folder_path):
        for img_name in files:

            if not img_name.lower().endswith(IMAGE_EXT):
                continue

            img_path = os.path.join(root, img_name)
            total_check += 1

            try:
                # 第一次打开：检查文件结构
                with Image.open(img_path) as img:
                    img.verify()

                # 第二次打开：尝试完整解码图片数据
                with Image.open(img_path) as img:
                    img.load()

            except Exception as e:
                broken_count += 1

                print(f"\n[发现损坏或无法读取的图片]")
                print(f"路径：{img_path}")
                print(f"原因：{e}")

                if not DRY_RUN:
                    try:
                        os.remove(img_path)
                        deleted_count += 1
                        print("处理结果：已删除")
                    except OSError as delete_error:
                        print(f"处理结果：删除失败，{delete_error}")
                else:
                    print("处理结果：仅检查，未删除")

    print(f"\n目录检查完成：{folder_path}")
    print(f"检查图片数量：{total_check}")
    print(f"发现异常图片：{broken_count}")
    print(f"成功删除图片：{deleted_count}\n")

    return broken_count, total_check, deleted_count

def augment_train_dataset(
        src_train_root: str,  # 形参1：原始训练集路径 flower/train
        dst_aug_root: str,  # 形参2：增强后图片保存路径 flower/train_aug
        aug_times: int = 2  # 形参3：扩充倍数，每张原图生成 aug_times 张增强图，默认值2
):
    """
    函数：对分类数据集训练集做离线数据增强，生成新图片保存到硬盘
    【函数输入】
        src_train_root: str，原始训练集根目录（文件夹内包含rose/tulip等类别子文件夹）
        dst_aug_root: str，增强后图片输出目录字符串
        aug_times: int，每张原图生成几张增强样本，默认=2
    【函数输出】
        无返回值，直接在硬盘创建文件夹、保存增强后的jpg图片
    """
    # ---------------------- 定义增强策略（可自由增删操作） ----------------------
    # transforms.Compose()：把多个图像变换操作打包，按先后顺序执行
    # 输入：transform操作组成的列表；输出：一个可调用的变换对象，传入图片就能执行全部变换
    aug_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5)：随机水平翻转
        # 输入参数p：翻转概率；输入图片PIL对象；输出翻转后的PIL图片
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=15)：随机旋转
        # 输入degrees：最大旋转角度±15°；输入PIL图片；输出旋转后图片
        transforms.RandomRotation(degrees=15),
        # transforms.ColorJitter：颜色抖动，随机修改亮度、对比度、饱和度、色调
        # 输入：brightness亮度，contrast对比度，saturation饱和度，hue色调；输入PIL图，输出修改颜色后的图
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # transforms.RandomResizedCrop：随机裁剪+缩放到固定尺寸
        # size=(224,224)输出图片大小；scale=(0.8,1.0)裁剪区域占原图比例范围；输入PIL图，输出裁剪缩放后的图片
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    ])
    # --------------------------------------------------------------------------

    # os.makedirs() 创建文件夹
    # 输入：路径字符串；exist_ok=True：文件夹已经存在时不会报错；无返回值，直接在硬盘建文件夹
    os.makedirs(dst_aug_root, exist_ok=True)

    # os.listdir(path)：读取指定文件夹下所有文件/子文件夹名称，返回字符串列表
    # 输入：文件夹路径src_train_root（原始训练集）；输出：该目录下所有文件名、文件夹名的列表
    for class_dir in os.listdir(src_train_root):
        # os.path.join() 拼接路径，自动适配Windows/Linux斜杠
        # 输入多个路径片段字符串；输出拼接完成的完整路径字符串
        src_class_path = os.path.join(src_train_root, class_dir)

        # os.path.isdir(path) 判断传入路径是不是文件夹
        # 输入路径字符串；输出布尔值：True是文件夹，False是文件
        if not os.path.isdir(src_class_path):
            # 如果不是文件夹，跳过本次循环，不处理
            continue

        # 拼接输出目录里对应类别的文件夹路径
        dst_class_path = os.path.join(dst_aug_root, class_dir)
        # 在输出目录新建对应类别文件夹，保持目录结构和原训练集一致
        os.makedirs(dst_class_path, exist_ok=True)

        # 遍历当前类别文件夹下所有图片文件名称
        for img_name in os.listdir(src_class_path):
            # 判断文件名后缀是否是jpg（.JPG大写也识别）
            # str.lower()：字符串全部转为小写；str.endswith() 判断结尾字符
            if not img_name.lower().endswith(".jpg"):
                continue
            # 拼接原图完整路径
            img_src_path = os.path.join(src_class_path, img_name)

            try:
                # Image.open() 打开图片文件
                # 输入图片路径字符串；输出PIL.Image图片对象；图片损坏会抛出异常
                img = Image.open(img_src_path).convert("RGB")
                # .convert("RGB")：PIL图片对象的成员方法，统一转为RGB三通道图片，避免RGBA透明通道报错
            except Exception as e:
                # 捕获图片打开失败异常（文件损坏、无法识别等），跳过这张图片，继续处理下一张
                print(f"【增强阶段跳过损坏图片】 {img_src_path}, 错误: {e}")
                continue

            # 【核心循环：一张原图循环生成 aug_times 张不同增强图片】
            for i in range(aug_times):
                # aug_transform(img)：执行上面Compose打包好的全部图像增强操作
                # 输入PIL图片对象；输出经过随机变换后的新PIL图片对象
                aug_img = aug_transform(img)

                # os.path.splitext(img_name) 分割文件名和后缀
                # 输入文件名字符串；输出元组 (文件名不带后缀, 后缀)
                name_without_ext = os.path.splitext(img_name)[0]
                # 构造新文件名：原图名字_增强编号.jpg，防止文件名重复覆盖
                new_img_name = f"{name_without_ext}_aug{i}.jpg"
                # 拼接增强图片的完整保存路径
                new_img_path = os.path.join(dst_class_path, new_img_name)

                # aug_img.save() PIL图片对象的成员方法，保存图片到硬盘
                # 输入保存路径字符串；无返回值，直接写入图片文件
                aug_img.save(new_img_path)
        # 打印当前类别处理完成提示
        print(f"类别 {class_dir} 增强完成")
    # 全部图片处理完毕，打印结束提示
    print("===== 全部训练集增强完毕！ =====")

# ============ 程序入口，扫描+增强调用示例（可按需打开注释运行） ============
if __name__ == "__main__":
    # 1. 先扫描并删除train、test里面损坏图片
    for sub_folder in SUB_FOLDERS:
        folder_full = os.path.join(DATA_ROOT, sub_folder)
        scan_broken_images(folder_full)

    # 2. 执行离线增强
    # augment_train_dataset(
    #     src_train_root=os.path.join(DATA_ROOT, "train"),
    #     dst_aug_root=os.path.join(DATA_ROOT, "train_aug"),
    #     aug_times=2
    # )


# -------------------完成调整图片大小以适应CNN输入-------------------
# transforms.Compose｜输入：transform变换操作列表；输出：组合好的图像变换对象
# 作用：将多个图像预处理操作打包，按顺序执行
train_transform = transforms.Compose([
    # transforms.Resize｜输入：目标尺寸元组(H,W)；输出：变换对象，对PIL图片缩放
    # 作用：调整图片尺寸，适配CNN网络输入，这里统一为224×224
    transforms.Resize((224, 224)),

    # ========== 在这里新增【训练集数据增强操作】==========
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转，50%概率翻转图片
    transforms.RandomRotation(degrees=15),    # 随机旋转 ±15度
    # =====================================================

    # transforms.ToTensor｜输入：PIL图像；输出：torch.Tensor张量
    # 作用：
    # 1.像素值从0~255映射到0~1
    # 2.维度从 (高度H,宽度W,通道C) → (通道C,高度H,宽度W)，CNN标准格式
    transforms.ToTensor(),
])
#------------------------------------------------------------------

# 测试集预处理：测试集**不能使用随机增强**，只做尺寸缩放+转张量
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()   #父类初始化
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #卷积层
        self.conv1 = Conv2d(in_channels=3,out_channels=96,kernel_size=11, stride=4,padding=0)
        self.conv2 = Conv2d(in_channels=96,out_channels=256,kernel_size=5, stride=1,padding=2)
        self.conv3 = Conv2d(in_channels=256,out_channels=384,kernel_size=3, stride=1,padding=1)
        self.conv4 = Conv2d(in_channels=384,out_channels=384,kernel_size=3, stride=1,padding=1)
        self.conv5 = Conv2d(in_channels=384,out_channels=256,kernel_size=3, stride=1,padding=1)
        #池化层
        self.pool = MaxPool2d(kernel_size=3, stride=2, padding=0)
        #全连接层
        self.fc6 = nn.Linear(in_features=6400, out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=18)
        #加入dropout，防止过拟合
        self.dropout = nn.Dropout(p=0.5)
        #激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # forward：数据前向传播流程，按图顺序执行
        # x：输入张量，shape = [batch_size, 3, 224, 224]，一批图片
        x = self.relu(self.conv1(x))
        '''
            1. self.conv1(x)：执行C1卷积（线性运算 Wx+b）
               输入图片提取底层特征：边缘、线条；输出shape [B,96,55,55]
               ⚠️卷积只是线性变换，只能学习直线类关系，多层卷积叠加等价单层
            2. self.relu( ... )：给卷积结果施加非线性激活
               把卷积输出里所有负数置0，保留正数。
               ✅核心目的：引入非线性，让网络有能力学习图片复杂特征（曲线、纹理、物体轮廓）
               如果不写relu，不管堆多少卷积，网络只能拟合线性关系，无法完成图像分类。
            执行完：x.shape = [B,96,55,55]
            '''
        x = self.pool(x)
        # -> C2 -> MaxPool
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # -> C3
        x = self.relu(self.conv3(x))
        # -> C4
        x = self.relu(self.conv4(x))
        # -> C5 -> MaxPool
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        # 展平：把特征图拉成一维向量，batch维度不动
        '''
        torch.flatten(x, start_dim=1)
        - start_dim=1：从第1维开始展平，保留第0维batch不变！
        原来x：[B, 256, 6, 6]
        把256×6×6这三个维度压成一维长向量：256*6*6=9216
        展平后shape：[B, 9216]
        ✅为什么要flatten？
        全连接层nn.Linear只能接收一维向量作为样本特征，不能接收二维特征图**，必须把H、W、通道全部拉直。
        '''
        x = torch.flatten(x, 1)
        # 全连接FC6 FC7 FC8
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        '''
            fc8：4096→18，最后一层，输出18个类别的原始分数(logits)
            ❗这里不加ReLU！！
            原因：后面损失函数CrossEntropyLoss内部自带softmax，需要原始分数，不能把负数清零。
            如果加ReLU，负的分类得分会被置0，分类结果出错。
            输出shape [B,18]
            '''
        x = self.fc8(x)
        return x


# ===================== 4. 训练主循环 =====================
# ===================== 4. 训练主循环 =====================
def train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    lr=1e-4
):
    """
    模型训练主函数：包含训练阶段、验证阶段、指标记录、绘图、保存权重
    Args:
        model: 实例化好的神经网络模型
        train_loader: 训练集DataLoader批次加载器
        test_loader: 测试集DataLoader批次加载器
        epochs: 训练总轮数
        lr: 学习率
    Returns:
        model: 训练完成后的模型
    """
    # 自动判断设备：优先cuda(GPU)，无GPU则使用cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # 打印当前程序使用的计算设备
    print(f"当前使用设备：{device}")

    # 将模型移动到指定设备(GPU/CPU)，模型后续计算都在该设备上执行
    model = model.to(device)

    # 损失函数：交叉熵损失，用于多分类任务，内部自带softmax计算
    criterion = nn.CrossEntropyLoss()

    # 优化器Adam，自适应学习率优化算法，用来更新网络参数
    optimizer = torch.optim.Adam(
        model.parameters(),   # 需要优化的参数：模型所有权重与偏置
        lr=lr                 # 设置学习率，控制参数更新步长
    )

    # 列表：保存每一轮epoch训练集准确率
    train_acc_list = []
    # 列表：保存每一轮epoch测试集准确率
    test_acc_list = []

    # 列表：保存每一轮epoch训练集损失
    train_loss_list = []
    # 列表：保存每一轮epoch测试集损失
    test_loss_list = []

    # 按epoch循环训练，一轮epoch代表完整遍历一遍全部训练数据
    for epoch in range(epochs):
        # =========================
        # 训练阶段
        # =========================
        # 设置模型为训练模式：启用Dropout；本网络无BN层，BN训练时会启用统计更新
        model.train()

        # 累计本轮训练总损失，初始化为0
        train_loss_sum = 0.0
        # 累计本轮训练预测正确样本数量
        train_correct = 0
        # 累计本轮训练样本总数
        train_total = 0

        # 迭代读取训练批次：images一批图片张量，labels对应图片的标签
        for images, labels in train_loader:
            # 将图片张量搬运到GPU/CPU设备
            images = images.to(device)
            # 将标签张量搬运到GPU/CPU设备
            labels = labels.to(device)

            # 梯度清零：pytorch梯度会自动累加，每轮batch开始前清空上一批梯度
            optimizer.zero_grad()

            # 前向传播：把图片送入模型，输出各类别logits原始得分
            outputs = model(images)
            # 计算当前批次预测结果与真实标签之间的损失
            loss = criterion(outputs, labels)

            # 反向传播：自动微分，计算损失对模型所有参数的梯度
            loss.backward()
            # 优化器利用梯度更新模型权重参数
            optimizer.step()

            # 获取当前批次样本数量
            batch_size = labels.size(0)
            # 累加总损失：loss是batch平均损失，乘batch_size得到该batch总损失
            train_loss_sum += loss.item() * batch_size

            # argmax(dim=1)：在类别维度上取最大值索引，作为预测类别
            predictions = outputs.argmax(dim=1)
            # 比较预测标签和真实标签，统计相等样本数量，转成python数字
            train_correct += (predictions == labels).sum().item()
            # 累加样本总数
            train_total += batch_size

        # 计算本轮训练集平均损失：总损失 / 全部样本数量
        train_loss = train_loss_sum / train_total
        # 计算本轮训练集准确率，转为百分比
        train_acc = 100.0 * train_correct / train_total

        # =========================
        # 测试阶段
        # =========================
        # 设置模型评估模式：关闭Dropout，BN层使用全局均值方差，不更新权重
        model.eval()

        # 累计本轮测试总损失
        test_loss_sum = 0.0
        # 累计本轮测试预测正确样本数量
        test_correct = 0
        # 累计本轮测试样本总数
        test_total = 0

        # 关闭梯度计算：不构建计算图，节省显存，加快推理速度
        with torch.no_grad():
            # 遍历测试集所有batch
            for images, labels in test_loader:
                # 图片张量移动到目标设备
                images = images.to(device)
                # 标签张量移动到目标设备
                labels = labels.to(device)

                # 前向传播，得到测试集预测logits
                outputs = model(images)
                # 计算测试批次损失
                loss = criterion(outputs, labels)

                # 获取当前batch样本数量
                batch_size = labels.size(0)
                # 累加测试集总损失
                test_loss_sum += loss.item() * batch_size
                # 获取预测类别
                predictions = outputs.argmax(dim=1)
                # 统计预测正确样本数
                test_correct += (predictions == labels).sum().item()
                # 累加测试样本总数
                test_total += batch_size

        # 计算本轮测试集平均损失
        test_loss = test_loss_sum / test_total
        # 计算本轮测试集准确率（百分比）
        test_acc = 100.0 * test_correct / test_total

        # =========================
        # 保存当前 Epoch 的指标
        # =========================
        # 将本轮训练准确率加入列表，后续绘图使用
        train_acc_list.append(train_acc)
        # 将本轮测试准确率加入列表
        test_acc_list.append(test_acc)
        # 将本轮训练损失加入列表
        train_loss_list.append(train_loss)
        # 将本轮测试损失加入列表
        test_loss_list.append(test_loss)

        # 打印本轮训练、测试指标
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.2f}%"
        )

    # =========================
    # 保存模型权重（所有epoch训练完成后，只保存最终模型）
    # =========================
    # model.state_dict() 获取模型所有参数（权重、偏置），保存到本地文件
    torch.save(
        model.state_dict(),
        "./my_alexnet_18cls.pth"
    )
    print("\n模型权重已保存。")

    # =========================
    # 绘制准确率变化曲线
    # =========================
    # 生成epoch横坐标序列：1,2,...,epochs
    epochs_range = range(1, epochs + 1)
    # 创建画布，设置画布大小
    plt.figure(figsize=(10, 6))

    # 绘制训练准确率曲线
    plt.plot(
        epochs_range,
        train_acc_list,
        marker="o",        # 线上圆点标记
        linewidth=2,       # 线条宽度
        label="Train Accuracy"
    )
    # 绘制测试准确率曲线
    plt.plot(
        epochs_range,
        test_acc_list,
        marker="s",        # 线上方块标记
        linewidth=2,
        label="Test Accuracy"
    )
    # X轴标签
    plt.xlabel("Epoch", fontsize=12)
    # Y轴标签
    plt.ylabel("Accuracy (%)", fontsize=12)
    # 图像标题
    plt.title(
        "Training and Testing Accuracy",
        fontsize=14
    )
    # 设置x轴刻度，每个epoch都显示
    plt.xticks(list(epochs_range))
    # Y轴范围固定0~100（百分比）
    plt.ylim(0, 100)
    # 打开网格线，虚线，透明度0.5
    plt.grid(True, linestyle="--", alpha=0.5)
    # 显示图例，区分训练/测试曲线
    plt.legend()
    # 自动调整布局，防止文字被截断
    plt.tight_layout()

    # 保存曲线图片到本地
    plt.savefig(
        "./accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("准确率曲线已保存至：accuracy_curve.png")

    # 在窗口弹出显示图片
    plt.show()

    # =========================
    # 可选：绘制损失变化曲线
    # =========================
    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制训练损失曲线
    plt.plot(
        epochs_range,
        train_loss_list,
        marker="o",
        linewidth=2,
        label="Train Loss"
    )
    # 绘制测试损失曲线
    plt.plot(
        epochs_range,
        test_loss_list,
        marker="s",
        linewidth=2,
        label="Test Loss"
    )
    # X轴名称
    plt.xlabel("Epoch", fontsize=12)
    # Y轴名称
    plt.ylabel("Loss", fontsize=12)
    # 图片标题
    plt.title("Training and Testing Loss", fontsize=14)
    # 设置x轴刻度
    plt.xticks(list(epochs_range))
    # 网格线
    plt.grid(True, linestyle="--", alpha=0.5)
    # 图例
    plt.legend()
    # 自适应布局
    plt.tight_layout()

    # 保存损失曲线图片
    plt.savefig(
        "./loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("损失曲线已保存至：loss_curve.png")
    # 弹出窗口展示图片
    plt.show()

    # 返回训练完成后的模型对象
    return model



# 程序入口：当这个py文件直接运行时，执行下面代码；被其他文件import导入时不会执行
#如果当前文件是**直接被当作主程序运行**，就执行下面代码；
#如果只是被别的文件导入（import），这一块代码不会自动跑。
if __name__ == "__main__":
    # 第一步：扫描并删除训练集、测试集损坏图片
    print("===== 开始扫描清理损坏图片 =====")
    total_del = 0
    total_all = 0
    for sub in SUB_FOLDERS:
        target_dir = os.path.join(DATA_ROOT, sub)
        broken_cnt, all_cnt, del_cnt = scan_broken_images(target_dir)
        total_del += del_cnt
        total_all += all_cnt
    print(f"全局扫描完成，总共检查{total_all}张图片，删除损坏图片{total_del}张\n")

    # ===== 【可选】如果你想用离线增强，取消下面两行注释，运行一次生成增强数据集 =====
    # source_train = os.path.join(DATA_ROOT, "train")
    # output_train_aug = os.path.join(DATA_ROOT, "train_aug")
    # augment_train_dataset(src_train_root=source_train, dst_aug_root=output_train_aug, aug_times=2)

    # 第二步：加载数据集（在线增强方案）
    train_dataset = ImageFolder(root=os.path.join(DATA_ROOT, "train"), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(DATA_ROOT, "test"), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 第三步：初始化模型 + 开始训练
    model = Mymodel()
    print("开始训练！")
    train_model(model, train_loader, test_loader, epochs=10, lr=1e-4)