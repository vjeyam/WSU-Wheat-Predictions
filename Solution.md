## The Process

Please note, that if you don't want to go through the entire process, you can [skip ahead](/README.md)!

The general solution can be seen in the image below:

![](/assets/image_processing.png "Image Processing")

An example of our data can be seen below:

![Image With Panel](/assets/image_with_panel.png)

or

![Image Without Panel](/assets/image_without_panel.png)

All of the images that didn't have a gray reference panel were intially in `data/cam8`, however I removed all of the images to improve the YOLOv8 detection model.

Before I did any image separation, I used a pretrained [YOLOv8 model](https://github.com/ultralytics/ultralytics) to detect the gray reference panel on both the RGB and NIR sides. We need to detect the gray reference panel so we can do radiometric reflection correction. Radiometric correction is done to calibrate the pixel values and correct for errors in the values. Note that it is not necessary to detect the gray reference panel before performing image separation.

In order to separate this image into a Multispectral - in this case called NIR images, we can run a simple function as shown below. You can run this function in the `notebooks/image_segmentation.ipynb` file.

``` python
def process_nir_images(input_folder, output_folder_ir):
    # List all camera folders
    cam_folders = [f'cam{i}' for i in range(1, 9)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_ir_path = os.path.join(output_folder_ir, f'{cam}_nir')
        
        os.makedirs(output_cam_ir_path, exist_ok=True)
        
        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Check the image size to ensure it is 1280 x 928
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                
                # Split the image to get the left half
                left_half = img.crop((0, 0, 640, 928))
                
                # Save the left half
                left_half.save(os.path.join(output_cam_ir_path, 
                f'{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}'))
                
                print(f"Processed left side of {filename}")

# Input and output folders
input_folder = '../data'
output_folder_ir = '../data/'

# Run the functions
process_nir_images(input_folder, output_folder_ir)
```

### Using a pretrained model

Using our original data I labeled all of the data through [Make Sense AI](https://www.makesense.ai/). Once done, I randomly selected 70% of the data to go to training, 20% to validating, and 10% to testing/predicting.
I trained the model on 200 epochs and got a mAP@0.50[1] of **99.5%** and a mAP@0.50-0.95[2] of **71.0%**. The testing accuracy ranged from **82%** to **85%** on all images.

Training and Validation:
![Training and Validation Image](/assets/panel/val/PR_curve.png)

![Results Image](/assets/panel/val/results.png)

Testing:
![Testing Image](/assets/panel/predict/cam8.png)

Once the detection model was completed, we need to locate the center (x,y) coordinates.

We created a script called [`detect.py`](model/panel/detect.py) to do this. Here is the general basis of what `detect.py` does:

*Please note that the code below is a extremely simplified version of what `detect.py` does!*

```python
# Process images from the input folder and save the processed images to the output folder
def process_images(input_folder, output_folder, centers_list):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for img_path in Path(input_folder).glob('*.png'):
        img = cv2.imread(str(img_path))
        results = model(img)
        
        for result in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, result[:4])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            centers_list.append((img_path.name, center_x, center_y, 5, 5))

        cv2.imwrite(str(Path(output_folder) / img_path.name), img)

# Crop and resize images from 1280 x 928 to 640 x 928
def crop_images(input_folder):
    for img_path in Path(input_folder).glob('*.png'):
        img = cv2.imread(str(img_path))
        cropped_img = img[:, :640]  # Crop the right side to get 640 x 928
        cv2.imwrite(str(img_path), cropped_img)

# Save center coordinates to CSV with additional columns 'Width' and 'Height'
def save_to_csv(centers_list, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Center_X', 'Center_Y', 'Width', 'Height'])
        writer.writerows(centers_list)
    print(f"Saved center coordinates to {output_csv}")
```

With these coordinates, we can create a smaller box inside the reference panel to calculate the red, green, and blue (RGB) values.







[1]: Mean Average Precision at IoU threshold 0.50.  
[2]: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05.