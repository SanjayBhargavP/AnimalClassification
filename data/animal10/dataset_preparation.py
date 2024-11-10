import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(root_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    class_names = os.listdir(root_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for class_name in class_names:
            class_split_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_split_dir):
                os.makedirs(class_split_dir)
    
    # Split dataset
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        images = os.listdir(class_dir)
        train_and_val_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
        train_images, val_images = train_test_split(train_and_val_images, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
        
        # Copy images to respective folders
        for img_name in train_images:
            shutil.copy(os.path.join(class_dir, img_name), os.path.join(output_dir, 'train', class_name, img_name))
        for img_name in val_images:
            shutil.copy(os.path.join(class_dir, img_name), os.path.join(output_dir, 'val', class_name, img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(class_dir, img_name), os.path.join(output_dir, 'test', class_name, img_name))
            
root_dir = './raw-img'
output_dir = './cooked_data'

prepare_dataset(root_dir=root_dir, output_dir=output_dir)