import os
import random
import cv2
import numpy as np
from captcha.image import ImageCaptcha

# 字符集：数字+大小写字母（62种）
# 修改captcha_generator.py中的CHAR_SET
CHAR_SET = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]  # 10数字+26小写=36类
# 重新生成样本→分割→训练，验证模型是否能收敛
CAPTCHA_LEN = 4  # 验证码长度（可调整）
CAPTCHA_HEIGHT = 60  # 验证码高度
CAPTCHA_WIDTH = 160  # 验证码宽度
PIC_DIR = "pic"
os.makedirs(PIC_DIR, exist_ok=True)

def generate_captcha(num_samples=1000):
    """生成验证码图片并保存"""
    image = ImageCaptcha(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT)
    for i in range(num_samples):
        # 随机生成4位验证码
        captcha_text = ''.join(random.choice(CHAR_SET) for _ in range(CAPTCHA_LEN))
        # 生成验证码图片
        img = image.generate_image(captcha_text)
        # 保存图片（文件名=验证码内容）
        img_path = os.path.join(PIC_DIR, f"{captcha_text}_{i}.png")
        img.save(img_path)
    print(f"已生成{num_samples}张验证码图片，保存至{PIC_DIR}")

def preprocess_image(img_path):
    """图像预处理：灰度化→二值化→降噪"""
    # 1. 读取图片
    img = cv2.imread(img_path)
    # 2. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 原固定阈值：_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # 改为自适应二值化（更鲁棒）
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # 4. 降噪（形态学开运算）
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return denoised

if __name__ == "__main__":
    # 生成1000张验证码图片
    generate_captcha(num_samples=1000)
    # 测试预处理（以第一张图片为例）
    test_img = os.listdir(PIC_DIR)[0]
    test_img_path = os.path.join(PIC_DIR, test_img)
    preprocessed = preprocess_image(test_img_path)
    # 保存预处理后的图片
    cv2.imwrite("test_preprocessed.png", preprocessed)
    print("预处理测试完成，生成test_preprocessed.png")