import os
from PIL import Image

# Parameters
input_dir = "results/gradcam_images/experiment_1"  # Path to the root folder containing Grad-CAM images
output_dir = "results/gradcam_gifs"  # Path to save the GIFs
duration = 2000 / 90  # Duration per frame (milliseconds). Adjust for 2 seconds total for 90 images.

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to create a GIF from images
def create_gif(image_folder, output_path):
    # List image files matching the expected naming convention
    images = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.startswith("age_") and f.endswith(".png")  # Filter valid files
        ],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])  # Extract age from filename
    )

    if not images:
        print(f"No valid images found in {image_folder}. Skipping GIF creation.")
        return

    # Open images and create a GIF
    frames = [Image.open(img) for img in images]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved GIF: {output_path}")

# Iterate through diagnosis and image folders
for diagnosis in os.listdir(input_dir):
    diagnosis_dir = os.path.join(input_dir, diagnosis)
    if not os.path.isdir(diagnosis_dir):
        continue

    # Create output directory for the diagnosis
    diagnosis_output_dir = os.path.join(output_dir, diagnosis)
    os.makedirs(diagnosis_output_dir, exist_ok=True)

    for image_id in os.listdir(diagnosis_dir):
        image_dir = os.path.join(diagnosis_dir, image_id)
        if not os.path.isdir(image_dir):
            continue

        # Create GIF for the current image ID
        output_path = os.path.join(diagnosis_output_dir, f"{image_id}.gif")
        create_gif(image_dir, output_path)
