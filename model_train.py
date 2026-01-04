import tensorflow as tf  
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from captcha_generator import CHAR_SET
from keras.layers import Input

DATASET_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 字符→标签映射
char_to_label = {char: idx for idx, char in enumerate(CHAR_SET)}
label_to_char = {idx: char for char, idx in char_to_label.items()}
NUM_CLASSES = len(CHAR_SET) 
IMG_SIZE = (28, 28)

def load_dataset():
    """加载标注好的字符数据集"""
    X = []
    y = []
    for char in CHAR_SET:
        char_dir = os.path.join(DATASET_DIR, char)
        if not os.path.exists(char_dir):
            continue
        for img_file in os.listdir(char_dir):
            img_path = os.path.join(char_dir, img_file)
            # 读取并归一化图片
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            # 扩展维度（适配CNN输入：(28,28)→(28,28,1)）
            img_array = np.expand_dims(img_array, axis=-1)
            X.append(img_array)
            # 标签编码
            y.append(char_to_label[char])
    # 转换为numpy数组
    X = np.array(X)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def build_model():
    """搭建CNN模型"""
    model = Sequential([
        # 输入层（适配Keras警告）
        Input(shape=(28, 28, 1)),
        # 卷积层1（增加滤波器数量）
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        # 卷积层2
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        # 卷积层3
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        # 展平+全连接层（增加神经元）
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.3),  # 降低dropout比例，减少信息丢失
        Dense(256, activation="relu"),
        Dropout(0.3),
       # 2. model_train.py中修改模型输出层：
        Dense(36, activation="softmax")  # 从62→36，匹配标签维度
    ])
    # 优化器调整学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
    # 编译模型
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model

def train_model():
    X_train, X_val, y_train, y_val = load_dataset()
    print(f"训练集数量：{len(X_train)}，验证集数量：{len(X_val)}")
    model = build_model()
    
    # 学习率调度：训练后期降低学习率
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001
    )
    
    # 增加epochs到50，添加回调函数
    history = model.fit(
        X_train, y_train,
        epochs=50,  # 从15→50
        batch_size=64,  # 从32→64，提升训练效率
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr]  # 加入学习率调度
    )
    
    # 保存为新版格式（解决HDF5警告）
    model_path = os.path.join(MODEL_DIR, "captcha_cnn_model.keras")
    model.save(model_path)
    print(f"模型已保存至{model_path}")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"验证集准确率：{val_acc:.4f}")

if __name__ == "__main__":
    train_model()