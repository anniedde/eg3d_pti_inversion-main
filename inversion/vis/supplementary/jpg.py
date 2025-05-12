import os
import cv2

def convert_png_to_jpg(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png') and 'grid' in file.lower():
                png_path = os.path.join(root, file)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'
                try:
                    img = cv2.imread(png_path)
                    cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    print(f"Converted {png_path} to {jpg_path}")
                except Exception as e:
                    print(f"Failed to convert {png_path}: {str(e)}")

# Usage
folder_path = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/supplementary'
convert_png_to_jpg(folder_path)
