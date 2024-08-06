import subprocess
import os

def run_process_image(img_path, age, output_path, neonatal=False):
    # Ensure the new_process_image.py script is in the same directory
    script_path = os.path.join(os.path.dirname(__file__), 'new_process_image.py')
    
    # Construct the command
    command = [
        'python',
        script_path,
        img_path,
        str(age),
        output_path
    ]
    
    if neonatal:
        command.append('--neonatal')
    
    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Error output: {e.stderr}")
        return False

if __name__ == "__main__":
    # Example usage
    img_path = "/home/philip-mattisson/Desktop/data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz"
    age = 35
    output_path = "/home/philip-mattisson/Desktop/data/V2Out"
    neonatal = False  # Set to True if the subject is neonatal
    
    success = run_process_image(img_path, age, output_path, neonatal)
    
    if success:
        print("Process completed successfully.")
    else:
        print("Process failed.")
