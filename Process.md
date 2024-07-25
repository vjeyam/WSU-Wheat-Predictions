## The Process

Please note, that if you don't want to go through the entire process, you can skip ahead!

The general solution can be seen in the image below:

![](/assets/image_processing.png "Image Processing")

An example of our data can be seen below:

![Example of the Given Data](/data/cam1/date_1-5-2024_10.0.11_1.png)

Before I did any image separation, I used a pretrained [YOLOv8 model](https://github.com/ultralytics/ultralytics) to detect the grey reference panel on both the RGB and NIR sides. We need to detect the grey reference panel so we can do radiometric reflection correction. Radiometric correction is done to calibrate the pixel values and correct for errors in the values. Note that it is not necessary to detect the grey reference panel before performing image separation.

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

Once completed, we can move onto the radiometric reflection corrections.