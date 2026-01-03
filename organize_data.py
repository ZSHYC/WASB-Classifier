import os
import shutil
from pathlib import Path

def organize_frames(base_path):
    """
    重新组织frames文件夹中的数据结构
    1. 按 game_1, game_2, game_3... 的形式重命名子文件夹
    2. 在每个子文件夹下创建Clip1文件夹
    3. 将所有jpg文件移动到Clip1/下
    4. 将Clip1中的jpg文件按 img_1, img_2, img_3... 形式重命名
    """
    base_path = Path(base_path)
    
    # 获取所有子文件夹列表
    subfolders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for idx, subfolder in enumerate(subfolders, start=1):
        # 新的文件夹名称
        new_folder_name = f"game_{idx}"
        new_folder_path = base_path / new_folder_name
        
        # 重命名文件夹
        print(f"重命名文件夹: {subfolder.name} -> {new_folder_name}")
        subfolder.rename(new_folder_path)
        
        # 创建Clip1文件夹
        clip1_path = new_folder_path / "Clip1"
        clip1_path.mkdir(exist_ok=True)
        
        # 获取所有jpg文件并按原名排序（确保重命名顺序一致）
        jpg_files = sorted([f for f in new_folder_path.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])
        
        # 将所有jpg文件移动到Clip1文件夹并重命名
        for jdx, jpg_file in enumerate(jpg_files, start=1):
            new_jpg_name = f"img_{jdx}.jpg"
            destination = clip1_path / new_jpg_name
            print(f"  移动并重命名 {jpg_file.name} -> Clip1/{new_jpg_name}")
            shutil.move(str(jpg_file), str(destination))
        
        # 删除新文件夹中除了Clip1外的所有其他文件和文件夹
        for item in new_folder_path.iterdir():
            if item.is_dir() and item.name != "Clip1":
                print(f"  删除文件夹: {item.name}")
                shutil.rmtree(item)
            elif item.is_file():
                print(f"  删除文件: {item.name}")
                item.unlink()
        
        print(f"完成处理: {new_folder_name}\n")

if __name__ == "__main__":
    # 设置基础路径
    # base_path = "datasets/3"
    base_path = r"D:\Personal\Desktop\qingzhen"
    
    # 确认路径存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
    else:
        # 执行组织操作
        organize_frames(base_path)
        print("数据结构转换完成！")