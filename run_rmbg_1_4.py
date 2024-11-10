
import os
from transformers import pipeline

folder_path = "rmbg/benchmark"
output_path = "rmbg/RMBG-1.4-Res"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Initialize the pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {image_path}...")

        # Create the mask and apply it to the image
        pillow_mask = pipe(image_path, return_mask=True)  # Outputs a pillow mask
        pillow_image = pipe(image_path)  # Applies mask on input and returns a pillow image

        # Save the processed image to the specified output folder
        output_file_path = os.path.join(output_path, f"output_{filename}")
        pillow_image.save(output_file_path)

        print(f"Saved processed image to {output_file_path}")
