import os
import cv2
import numpy as np
from captcha_generator import preprocess_image, CHAR_SET, CAPTCHA_LEN, PIC_DIR

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# 为每个字符创建标注文件夹
for char in CHAR_SET:
    os.makedirs(os.path.join(DATASET_DIR, char), exist_ok=True)

def segment_characters(img_path, captcha_text):
    """分割验证码字符并标注保存（修复轮廓筛选+补全保存逻辑）"""
    # 预处理图片
    preprocessed = preprocess_image(img_path)
    # 兼容OpenCV 3/4版本的轮廓返回值（cv2.findContours返回值不同）
    contours, hierarchy = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选有效轮廓（调整阈值适配60x160验证码）
    char_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 调整轮廓筛选阈值：适配60（高）x160（宽）的验证码
        if 5 < w < 50 and 20 < h < 60:  # 放宽宽度阈值，避免过滤有效字符
            char_contours.append((x, y, w, h))
    
    # 按x坐标排序（保证字符顺序）
    char_contours = sorted(char_contours, key=lambda x: x[0])
    # 若轮廓数不足4个，降级为均分分割（兜底方案）
    if len(char_contours) < 4:
        char_width = preprocessed.shape[1] // CAPTCHA_LEN
        char_height = preprocessed.shape[0]
        char_contours = []
        for i in range(CAPTCHA_LEN):
            x = i * char_width
            y = 0
            w = char_width
            h = char_height
            char_contours.append((x, y, w, h))
    
    # 截取并保存字符（补全核心保存逻辑）
    for i, (x, y, w, h) in enumerate(char_contours[:4]):  # 取前4个轮廓（4位验证码）
        # 防止索引越界（验证码文本长度不足）
        if i >= len(captcha_text):
            continue
        # 截取单个字符区域（防止越界）
        char_img = preprocessed[max(0, y):min(y+h, preprocessed.shape[0]), 
                                max(0, x):min(x+w, preprocessed.shape[1])]
        # 标准化字符尺寸（28x28，适配CNN输入）
        char_img = cv2.resize(char_img, (28, 28))
        # 获取字符标注（当前位置的字符）
        char_label = captcha_text[i]
        # 生成保存路径（避免重名）
        save_filename = f"{captcha_text}_{os.path.basename(img_path).split('_')[1].split('.')[0]}_{i}.png"
        save_path = os.path.join(DATASET_DIR, char_label, save_filename)
        # 保存字符图片
        cv2.imwrite(save_path, char_img)

def create_dataset():
    """批量分割+标注数据集（增加日志+异常处理）"""
    count = 0
    success_count = 0
    # 检查pic文件夹是否有图片
    pic_files = [f for f in os.listdir(PIC_DIR) if f.endswith(".png")]
    if len(pic_files) == 0:
        print(f"错误：{PIC_DIR}文件夹中无png格式的验证码图片！")
        return
    
    for img_file in pic_files:
        img_path = os.path.join(PIC_DIR, img_file)
        # 从文件名提取验证码文本（格式：abcd_123.png → abcd）
        try:
            captcha_text = img_file.split("_")[0]
            # 验证验证码文本长度（必须是4位）
            if len(captcha_text) != CAPTCHA_LEN:
                print(f"跳过{img_file}：验证码文本长度不是{CAPTCHA_LEN}位")
                continue
            # 分割字符并保存
            segment_characters(img_path, captcha_text)
            success_count += 1
        except Exception as e:
            print(f"处理{img_file}失败：{e}")
        count += 1
    print(f"处理完成：共扫描{count}张图片，成功分割{success_count}张，数据集保存至{DATASET_DIR}")

if __name__ == "__main__":
    create_dataset()