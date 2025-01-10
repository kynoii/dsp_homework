import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置中文字体（根据你的系统和字体情况调整）
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取彩色图像
img = cv2.imread('pic/8.jpg')  # 修改为你的文件名,将pic/后面的部分更改即可
k = 1000  # 截断到第 k 个特征值

# 获取图像的尺寸
W, H = img.shape[0], img.shape[1]

# 将图像转换为浮点数矩阵,归一化到 [0, 1]
img_float32 = np.float32(img) / 255.0

# 分离RGB通道
red_channel = img_float32[:, :, 0]
green_channel = img_float32[:, :, 1]
blue_channel = img_float32[:, :, 2]

# 计算原始存储空间
original_storage = W * H * 3  # 3通道，RGB


# 处理每个通道
def process_channel(channel, k):
    # 计算奇异值分解
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)

    # 截断到前 k 个奇异值
    S_truncated = np.diag(S[:k])
    channel_reconstructed = np.dot(U[:, :k], np.dot(S_truncated, Vt[:k, :]))

    return channel_reconstructed, S[:k]


# 对每个通道进行处理
blue_reconstructed, blue_singular_values = process_channel(blue_channel, k)
green_reconstructed, green_singular_values = process_channel(green_channel, k)
red_reconstructed, red_singular_values = process_channel(red_channel, k)

# 重构图像
reconstructed_img = cv2.merge([blue_reconstructed, green_reconstructed, red_reconstructed])

reconstructed_img = np.uint8(reconstructed_img * 255)

# 计算压缩后的存储空间
compressed_storage = 3 * ((W * k) + (k * k) + (k * H))

# 计算压缩比
compression_ratio = original_storage / compressed_storage

# 绘制图像
plt.figure(figsize=(10, 5))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
plt.title(f'原图', fontsize=16)

# 重构图像
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(reconstructed_img))  # 显示彩色图像
plt.title(f'特征子图 1 至 {k} 累加图', fontsize=16)

plt.suptitle(f'矩阵阶数n= {W}, 压缩比= {compression_ratio:.3f}', fontsize=16)
plt.show()

# 打印每个通道的奇异值
print("蓝色通道奇异值：", blue_singular_values)
print("绿色通道奇异值：", green_singular_values)
print("红色通道奇异值：", red_singular_values)
