import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from captcha_generator import preprocess_image, CAPTCHA_LEN, CHAR_SET
from model_train import MODEL_DIR, IMG_SIZE

# 重建与训练时完全一致的标签映射表
char_to_label = {char: idx for idx, char in enumerate(CHAR_SET)}
label_to_char = {idx: char for char, idx in char_to_label.items()}

def segment_single_captcha(img_path):
    """分割单张验证码图片的字符"""
    preprocessed = preprocess_image(img_path)
    char_width = preprocessed.shape[1] // CAPTCHA_LEN
    char_imgs = []
    for i in range(CAPTCHA_LEN):
        char_img = preprocessed[:, i*char_width : (i+1)*char_width]
        char_img = cv2.resize(char_img, IMG_SIZE)
        char_img = np.array(char_img) / 255.0
        char_img = np.expand_dims(char_img, axis=-1)
        char_imgs.append(char_img)
    return np.array(char_imgs)

def recognize_captcha(img_path, model):
    """识别验证码"""
    char_imgs = segment_single_captcha(img_path)
    pred_labels = model.predict(char_imgs, verbose=0)  # 关闭预测进度条
    pred_chars = []
    for label in pred_labels:
        pred_idx = np.argmax(label)
        pred_char = label_to_char.get(pred_idx, "")
        pred_chars.append(pred_char)
    captcha_result = ''.join(pred_chars)
    return captcha_result, char_imgs

if __name__ == "__main__":
    # 加载模型
    model_path = os.path.join(MODEL_DIR, "captcha_cnn_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "captcha_cnn_model.h5")
    model = load_model(model_path)
    
    # 验证模型输出维度与字符集匹配
    model_output_dim = model.output_shape[-1]
    char_set_len = len(CHAR_SET)
    if model_output_dim != char_set_len:
        raise ValueError(f"模型输出维度{model_output_dim}≠字符集数量{char_set_len}")
    
    # 选择测试图片
    test_img_dir = "pic"
    test_img_files = [f for f in os.listdir(test_img_dir) if f.endswith(".png")][-10:]  # 最后10张测试
    
    # 统计指标初始化
    total_chars = 0  # 总单字符数
    correct_chars = 0  # 正确单字符数
    correct_captchas = 0  # 正确完整验证码数
    total_captchas = len(test_img_files)  # 总验证码数

    # 批量测试
    print("="*60)
    print("验证码识别结果（单字符维度）")
    print("="*60)
    print(f"{'真实值':<8} {'预测值':<8} {'单字符匹配情况'}")
    print("-"*60)

    for img_file in test_img_files:
        img_path = os.path.join(test_img_dir, img_file)
        true_captcha = img_file.split("_")[0]
        
        # 识别验证码
        pred_captcha, char_imgs = recognize_captcha(img_path, model)
        
        # 修复：fillchar改为单个空格（符合ljust参数要求）
        # 补全字符长度，超出则截断，不足则补空格
        true_captcha = true_captcha.ljust(CAPTCHA_LEN, " ")[:CAPTCHA_LEN]
        pred_captcha = pred_captcha.ljust(CAPTCHA_LEN, " ")[:CAPTCHA_LEN]
        
        # 统计单字符正确数
        char_match = []
        for t_char, p_char in zip(true_captcha, pred_captcha):
            total_chars += 1
            # 忽略空格（补全的占位符）
            if t_char == " " or p_char == " ":
                char_match.append("-")
                continue
            if t_char == p_char:
                correct_chars += 1
                char_match.append("√")
            else:
                char_match.append("×")
        
        # 统计完整验证码正确数（去除空格后对比）
        if true_captcha.strip() == pred_captcha.strip():
            correct_captchas += 1
        
        # 输出单字符匹配详情（去除空格，保持美观）
        true_captcha_show = true_captcha.strip()
        pred_captcha_show = pred_captcha.strip()
        match_str = " ".join(char_match)
        print(f"{true_captcha_show:<8} {pred_captcha_show:<8} {match_str}")
        
        # 保存分割后的字符图片（可选）
        for i, char_img in enumerate(char_imgs):
            char_img = (char_img * 255).astype(np.uint8)
            cv2.imwrite(f"char_{img_file}_{i}.png", char_img)

    # 计算准确率
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    captcha_accuracy = correct_captchas / total_captchas if total_captchas > 0 else 0

    # 输出最终统计结果
    print("="*60)
    print(f"【统计结果】")
    print(f"总验证码数：{total_captchas} | 完整验证码正确数：{correct_captchas} | 完整验证码准确率：{captcha_accuracy:.4f}")
    print(f"总单字符数：{total_chars} | 单字符正确数：{correct_chars} | 单字符准确率：{char_accuracy:.4f}")
    print("="*60)