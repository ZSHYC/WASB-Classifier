import os
import json

def delete_json_files_in_subfolders(root_folder):
    """
    删除指定文件夹下所有子文件夹中的JSON文件
    
    Args:
        root_folder: 根文件夹路径
    """
    # 遍历根文件夹下的所有子文件夹
    for root, dirs, files in os.walk(root_folder):
        # 遍历当前文件夹中的所有文件
        for file in files:
            # 检查文件是否为JSON文件
            if file.lower().endswith('.json'):
                json_file_path = os.path.join(root, file)
                try:
                    # 删除JSON文件
                    os.remove(json_file_path)
                    print(f"已删除: {json_file_path}")
                except Exception as e:
                    print(f"删除失败 {json_file_path}: {e}")

# 使用示例
if __name__ == "__main__":
    # 请将 "1/" 替换为您实际的文件夹路径
    folder_path = r"D:\Personal\Desktop\WASB-SBDT\datasets\tennis_predict"  # 您的文件夹路径
    delete_json_files_in_subfolders(folder_path)