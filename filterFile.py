import os
import re
import shutil
from PIL import Image, ImageFilter

def resize_and_copy_images(folder_path, output_folder):
    file_list = os.listdir(folder_path)

    new_file_list = []
    for filename in file_list:
        full_path = os.path.join(folder_path, filename)

        if os.path.isfile(full_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            new_filename = re.sub(r'\d+', '', filename)  # 去除文件名中的数字

            output_path = os.path.join(output_folder, new_filename)

            # 调整图片尺寸为 60x60
            img = Image.open(full_path)
            resized_img = img.resize((30, 30), Image.BILINEAR)  # 或者使用 Image.LANCZOS
            resized_img.save(output_path)  # 保存调整尺寸后的图片

            new_file_list.append(new_filename)

    return new_file_list


# 示例用法
folder_path = 'images/120'  # 替换为你的文件夹路径
output_folder = 'images/label'  # 替换为你想要保存修改后文件的目标文件夹路径

new_file_list = resize_and_copy_images(folder_path, output_folder)
print(new_file_list)