import subprocess
import os
import argparse

def run_process_image_with_options(img_path, age, output_path, neonatal=False, theta_x=0, theta_y=0, theta_z=0,
                                   conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625,
                                   threshold_filter="Otsu", mip_slices=5):
    script_path = os.path.join(os.path.dirname(__file__), 'new_process_image_with_options.py')
    
    # Construct the command
    command = [
        'python',
        script_path,
        img_path,
        str(age),
        output_path,
        '--theta_x', str(theta_x),
        '--theta_y', str(theta_y),
        '--theta_z', str(theta_z),
        '--conductance_parameter', str(conductance_parameter),
        '--smoothing_iterations', str(smoothing_iterations),
        '--time_step', str(time_step),
        '--threshold_filter', threshold_filter,
        '--mip_slices', str(mip_slices)
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
    parser = argparse.ArgumentParser(description="Run brain circumference calculation with options.")
    parser.add_argument("--img_path", type=str, default="/home/philip-mattisson/Desktop/data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz", help="Path to the MRI image file")
    parser.add_argument("--age", type=int, default=35, help="Age of the subject")
    parser.add_argument("--output_path", type=str, default="/home/philip-mattisson/Desktop/data/V2Out", help="Path to the output folder")
    parser.add_argument("--neonatal", action="store_false", help="Flag to indicate if the subject is neonatal")
    parser.add_argument("--theta_x", type=float, default=0, help="Rotation angle around x-axis")
    parser.add_argument("--theta_y", type=float, default=0, help="Rotation angle around y-axis")
    parser.add_argument("--theta_z", type=float, default=0, help="Rotation angle around z-axis")
    parser.add_argument("--conductance_parameter", type=float, default=3.0, help="Conductance parameter for anisotropic diffusion")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of smoothing iterations")
    parser.add_argument("--time_step", type=float, default=0.0625, help="Time step for anisotropic diffusion")
    parser.add_argument("--threshold_filter", type=str, default="Otsu", choices=["Otsu", "Binary"], help="Threshold filter method")
    parser.add_argument("--mip_slices", type=int, default=5, help="Number of slices for maximum intensity projection")
    
    args = parser.parse_args()

    success = run_process_image_with_options(
        args.img_path,
        args.age,
        args.output_path,
        args.neonatal,
        args.theta_x,
        args.theta_y,
        args.theta_z,
        args.conductance_parameter,
        args.smoothing_iterations,
        args.time_step,
        args.threshold_filter,
        args.mip_slices
    )
    
    if success:
        print("Process completed successfully.")
    else:
        print("Process failed.")
