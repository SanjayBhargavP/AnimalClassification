import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(root_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    cat_dir = 'raw-img/cat/'
    dog_dir = 'raw-img/dog/'

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    with open("list.txt", 'r') as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.split()
            image_name = parts[0]
            
            # Determine if it's a cat or dog based on the first character
            print(parts)
            print(parts[2])
            if parts[2] == "1":
                dest_dir = cat_dir
                print("cat")
            else:
                print("dog")
                dest_dir = dog_dir

            # Define source and destination paths
            src_path = os.path.join(root_dir, image_name + ".jpg")  # Assuming images are .jpg
            dest_path = os.path.join(dest_dir, image_name + ".jpg")

            # # Move the image to the appropriate directory
            if os.path.exists(src_path):  # Check if the source image exists
                shutil.move(src_path, dest_path)
                print(f"Moved {image_name} to {dest_dir}")
            else:
                print(f"File {src_path} does not exist.")
    
    class_names = ["dog", "cat"]
    
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