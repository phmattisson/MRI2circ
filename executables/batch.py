import argparse
import process_image


import argparse
import process_image  # Assuming this module has a function `main` that does image processing

def main(img_path, age, output_path):
    # Assuming the process_image.main function accepts these three arguments
    results = process_image.main(img_path, age, output_path)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate brain perimeter from MRI image.")
    parser.add_argument("--img_path", type=str, default='/Users/philipmattisson/Desktop/Centile/data/T1_UKB5563485.nii.gz', help="Path to the MRI image file")
    parser.add_argument("--age", type=int, default=35, help="Age of the subject")
    parser.add_argument("--output_path", type=str, default='/Users/philipmattisson/Desktop/Centile/MRI2circ/test_output/', help="Path to the output folder")

    args = parser.parse_args()
    
    # Ensure we are passing the arguments to the main function
    result = main(args.img_path, args.age, args.output_path)
    if result:
        # Ensure your processing results return a dictionary with these keys
        print(f'perimeter_opencv {result[0]}, perimeter_convex {result[1]}')
    else:
        print("Failed to process image.")