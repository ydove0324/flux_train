import os
import json

def create_jsonl_for_images(folder_path, prompt, output_file):
    """
    为指定文件夹中的所有图片创建JSONL格式数据
    
    参数:
    folder_path: 包含图片的文件夹路径
    prompt: 所有图片使用的统一提示词
    output_file: 输出的JSONL文件路径
    """
    # 支持的图片扩展名
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历文件夹中的所有文件
        for filename in sorted(os.listdir(folder_path)):
            # 检查文件是否为图片
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # 创建JSON条目
                entry = {
                    "file_name": filename,
                    "prompt": prompt
                }
                # 写入JSONL文件
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"JSONL文件已创建: {output_file}")
    print(f"共处理 {len([f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in image_extensions)])} 张图片")

# 使用示例
if __name__ == "__main__":
    # 参数设置
    folder_path = "Sofa_test1"  # 图片文件夹路径
    prompt = "Beige single sofa with curved design"  # 统一的提示词
    output_file = "sofa_dataset.jsonl"  # 输出文件名
    
    create_jsonl_for_images(folder_path, prompt, output_file)p