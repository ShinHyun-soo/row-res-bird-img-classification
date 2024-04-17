import cv2
import os
import pandas as pd
import imgaug.augmenters as iaa
from tqdm import tqdm

# Define your image directory and CSV file path
image_dir = "./train"
csv_path = "lowaug2.csv"

# Define augmentation pipeline
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Affine(rotate=(-10, 10)),  # rotate images by -10 to 10 degrees
    iaa.GaussianBlur(sigma=(0, 1.0)),  # apply gaussian blur with a sigma between 0 and 1.0
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # add gaussian noise to images
    iaa.Multiply((0.8, 1.2)),  # multiply image values by random values between 0.8 and 1.2
    iaa.Crop(percent=(0, 0.1))  # crop images by 0-10% of their height/width
])


# Function to read images from directory and generate augmented images
def augment_images(image_dir, csv_path):
    data = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_dir, filename)
            label = filename.split("_", 1)[1].split(".")[0].replace("_", " ")  # Extract label from filename
            img = cv2.imread(img_path)

            # Perform data augmentation
            augmented_img = augmentation(image=img)
            augmented_img_path = os.path.join(image_dir, "augmented_" + filename)
            cv2.imwrite(augmented_img_path, augmented_img)

            # Append augmented image path and label to data list
            data.append((augmented_img_path, label))

    # Convert data list to pandas DataFrame
    df = pd.DataFrame(data, columns=['img_path', 'label'])

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print("CSV file saved successfully.")


# Call the function to perform data augmentation and generate CSV
augment_images(image_dir, csv_path)