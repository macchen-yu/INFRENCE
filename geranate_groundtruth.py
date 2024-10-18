import numpy as np
import os
import argparse

# 你的字典
fabric_dict = {
    1: "T100.npy",
    2: "C100.npy",
    3: "C20T80.npy",
    4: "C80T20.npy",
    5: "C65T35.npy",
    6: "C35T65.npy",
    7: "C50T50.npy"
}

# 使用 argparse 來處理命令行參數
parser = argparse.ArgumentParser(description="Generate groundtruth files from specified folder path.")
parser.add_argument('folder_path', type=str, help='Path to the folder containing the .npy files')
args = parser.parse_args()

# 資料夾位置
folder_path = args.folder_path

# 創建 groundtruth 資料夾（如果不存在）
groundtruth_folder_path = os.path.join(folder_path, "groundtruth")
if not os.path.exists(groundtruth_folder_path):
    os.makedirs(groundtruth_folder_path)

# 遍歷資料夾內的檔案
for filename in os.listdir(folder_path):
    # 檢查檔名是否在字典值中
    for key, value in fabric_dict.items():
        if filename == value:
            print(f"檔案名 {filename} 匹配到字典中的 key: {key}")

            # 讀取匹配的三維 npy 檔案
            file_path = os.path.join(folder_path, filename)
            original_array = np.load(file_path)

            # 獲取長和寬
            height, width = original_array.shape[:2]

            # 創建一個與原始三維陣列長寬相同的二維 ndarray，值全部為指定的 key，並去掉頭尾各 3
            groundtruth_array = np.full((height - 6, width - 6), key)

            # 保存創建的二維陣列為新的 npy 檔案到 groundtruth 資料夾
            save_path = os.path.join(groundtruth_folder_path, f"gt_{filename}")
            np.save(save_path, groundtruth_array)

            # 輸出結果
            print(f"創建的二維陣列已保存至：{save_path}")
