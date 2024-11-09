import os
import pandas as pd
from natsort import natsorted


def generate_meta_data_jsonl(image_dir, label_dir, save_path, image_suffix='.bmp', label_suffix='.bmp'):
    image_filenames = [filename for filename in natsorted(os.listdir(image_dir)) if filename.endswith(image_suffix)]
    label_filenames = [filename for filename in natsorted(os.listdir(label_dir)) if filename.endswith(label_suffix)]

    assert image_filenames == label_filenames, "Image does not match label!"
    
    image_files = [os.path.join(image_dir, filename) for filename in image_filenames]
    label_files = [os.path.join(label_dir, filename) for filename in label_filenames]

    meta_data = pd.DataFrame({'img': image_files, 'seg':label_files})
    meta_data.to_json(save_path, orient='records', lines=True)


if __name__ == '__main__':
    image_dir = '/Users/yxizhong/workspace/DMCode/Dataset/RawPic4_val'
    label_dir = '/Users/yxizhong/workspace/DMCode/Dataset/Label4_val'
    save_path = '/Users/yxizhong/workspace/visual-tuning/lora/mmseg_datasets/val.jsonl'
    
    generate_meta_data_jsonl(image_dir, label_dir, save_path)
