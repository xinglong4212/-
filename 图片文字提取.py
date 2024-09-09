
# 增强图像清晰度的示例代码：
import cv2
import numpy as np

import cv2
import pytesseract
import pandas as pd
from pytesseract import Output
from openpyxl import Workbook

# 设置 tesseract 的路径 (如果需要)
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract'


def enhance_image(img):
    # 调整对比度
    contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    # 应用高斯模糊去噪
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)

    # 尝试锐化图像
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    return sharpened


# 图像预处理示例（去噪声 + 二值化）
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用高斯滤波去除噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 自适应阈值
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


def process_image(image_path):
    # 使用 OpenCV 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值进行二值化
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 使用 pytesseract 进行 OCR 识别
    details = pytesseract.image_to_data(thresh, output_type=Output.DICT)

    # 提取 OCR 结果中的表格结构
    rows = []
    row = []
    last_word_position = 0
    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 30:  # 仅使用可信度高的文本
            word = details['text'][i]
            word_position = details['top'][i]

            # 检查是否需要换行
            if abs(word_position - last_word_position) > 10 and row:
                rows.append(row)
                row = []
            
            row.append(word)
            last_word_position = word_position

    if row:
        rows.append(row)

    # 将内容保存到 Pandas DataFrame 中
    df = pd.DataFrame(rows)
    return df

def save_to_excel(df, output_path):
    # 保存 DataFrame 到 Excel 文件
    df.to_excel(output_path, index=False, header=False)

if __name__ == "__main__":
    # 图像路径
    image_path = 'picture_path'

    
    # 输出文件路径
    output_excel = 'output_table.xlsx'
    
    # 处理图像并保存为表格
    df = process_image(image_path)
    save_to_excel(df, output_excel)

    print(f"表格已保存到 {output_excel}")
