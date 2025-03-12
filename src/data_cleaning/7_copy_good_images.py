import pandas as pd
import shutil
import os

def copy_good_images():
    # Create final_data directory if it doesn't exist
    os.makedirs('data/final_data', exist_ok=True)
    
    # Read the annotations CSV file
    df = pd.read_csv('data/refined_data/annotations_cleaned.csv')
    
    # Filter only the good images
    good_images = df[df['annotation'] == 'Good']['image_name']
    
    # Copy each good image to the final_data folder
    for image_name in good_images:
        source_path = f'data/refined_data/{image_name}'
        destination_path = f'data/final_data/{image_name}'
        
        try:
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {image_name}")
        except FileNotFoundError:
            print(f"Warning: Could not find {image_name}")
        except Exception as e:
            print(f"Error copying {image_name}: {str(e)}")
    
    print(f"\nTotal good images copied: {len(good_images)}")

if __name__ == "__main__":
    copy_good_images()
