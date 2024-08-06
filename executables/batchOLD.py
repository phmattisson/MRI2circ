import argparse
import process_image

def main(img_path, age, output_path, neonatal):
    # Now passing the neonatal argument to process_image.main
    results = process_image.main(img_path, age, output_path, neonatal)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate brain perimeter from MRI image.")
    parser.add_argument("--img_path", type=str, default='/home/philip-mattisson/Desktop/data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz', help="Path to the MRI image file")
    parser.add_argument("--age", type=int, default=35, help="Age of the subject")
    parser.add_argument("--output_path", type=str, default='/home/philip-mattisson/Desktop/data/testOutput', help="Path to the output folder")
    parser.add_argument("--neonatal", action="store_true", help="Flag to indicate if the subject is neonatal")

    args = parser.parse_args()
    
    # Passing all arguments, including neonatal, to the main function
    result = main(args.img_path, args.age, args.output_path, args.neonatal)
    if result:
        # Ensure your processing results return a dictionary with these keys
        print(f'perimeter_opencv {result[0]}, perimeter_convex {result[1]}')
    else:
        print("Failed to process image.")