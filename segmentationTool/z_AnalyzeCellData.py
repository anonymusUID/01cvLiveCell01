#!C:\Users\GOURAV ROY\Desktop\All Folders\Integrated Workflows\Somatic Cell Detection\CellMasking_FuzzyLogic\.venv\Scripts\python.exe

import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
import os
import sys

#def analyze_and_label_objects(image, selected_features, min_area=10, mode ="excel", font_scale=0.5, font_color=(0, 255, 255), font_thickness=2):



def analyze_and_label_objects(image_path, out_dir, output_file_suff, selected_features, base_name, min_area=10, mode="excel", font_scale=0.5, font_color=(255, 0, 0), font_thickness=2):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return -1
    """
    # Now you can work with the image
    annotated_image = image.copy()
    bounding_box_image = image.copy()
    detected_contour_image = image.copy()
    """
    
    # Rest of your code...
    if mode not in ["excel", "console"]:
        print("Enter Valid Mode for function")
        return -1

    
    annotated_image = image.copy()
    bounding_box_image = image.copy()
    detected_contour_image = image.copy()
    
    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 1:
        gray = annotated_image[:, :, 0]
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    elif len(annotated_image.shape) == 2:
        gray = annotated_image
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    elif len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format: Ensure the input is grayscale or BGR color image.")
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_count = 0
    total_area = gray.shape[0] * gray.shape[1]
    centroids = []
    
    feature_map = {
        "General": ["Count", "Cell Density"],
        "Geometry": ["Perimeter", "Area", "Aspect Ratio"],
        "Morphology": ["Roundness", "Shape", "Convexity"],
        "Localization": ["2Dcoordinates", "Centroid"],
        "Orientation": ["Orientation"],
        "Voronoi Entropy": ["Voronoi_Entropy"]
    }
    
    selected_columns = ["ID"]
    selected_columns2 = []
    for key in selected_features.split(","):
      key = key.strip()   
      if key != "General" and key !="Voronoi Entropy":  
        if key in feature_map:
            selected_columns.extend(feature_map[key])
      else:
        selected_columns2.extend(feature_map[key])
    
   
    
    cell_data = {col: [] for col in selected_columns}
    cell_data2 = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        object_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(contour, True)
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        shape = "Undetermined"
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
        elif len(approx) > 5:
            shape = "Elliptical" if roundness > 0.75 else "Other"
        
        M = cv2.moments(contour)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
        centroids.append((cx, cy))
        
        mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
        angle = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) if (mu20 - mu02) != 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0  # Convexity ratio

        top_left = tuple(contour[contour[:, :, 1].argmin()][0])
        
        cv2.putText(annotated_image, f"{object_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(detected_contour_image, [contour], -1, (255, 0, 0), 2)
        
        cell = {"ID": object_count}
        if "Area" in selected_columns:
            cell["Area"] = area
        if "Aspect Ratio" in selected_columns:
            cell["Aspect Ratio"] = aspect_ratio
        if "Roundness" in selected_columns:
            cell["Roundness"] = roundness
        if "Shape" in selected_columns:
            cell["Shape"] = shape
        if "Centroid" in selected_columns:
            cell["Centroid"] = (cx, cy)
        if "Orientation" in selected_columns:
            cell["Orientation"] = angle
        if "Perimeter" in selected_columns:
            cell["Perimeter"] = perimeter
        if "Convexity" in selected_columns:
            cell["Convexity"] = convexity
        if "2Dcoordinates" in selected_columns:
            cell["2Dcoordinates"] = top_left
        
        for key in cell:
            cell_data[key].append(cell[key])
    
    concentration = object_count / total_area
    entropy_value = shannon_entropy(binary)
    
    if "Cell Density" in selected_columns2:
        cell_data2["Cell Density"] = concentration  # Store a single value
    if "Count" in selected_columns2:
        cell_data2["Count"] = object_count  # Store a single value
    if "Voronoi_Entropy" in selected_columns2:
        cell_data2["Voronoi_Entropy"] = float(entropy_value)  # Store a single value
    
    print(cell_data2)

    
    
    max_length = max(len(lst) for lst in cell_data.values())
    for key in cell_data.keys():
        while len(cell_data[key]) < max_length:
            cell_data[key].append(None)
    
    if mode == "console":
        for i in range(object_count):
            print(f"Object {cell_data['ID'][i]}:")
            for key in selected_columns[1:]:
                print(f"  {key}: {cell_data[key][i]}")
    elif mode == "excel":
       
        excel_filename = output_file_suff+".xlsx"
        
        df = pd.DataFrame(cell_data)
        df.to_excel(excel_filename, index=False)
    
    print(f"Saved Data to {excel_filename} Successfully", out_dir+output_file_suff+"_annotated.jpg")
    
    plt.figure(figsize=(30, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(annotated_image)
    plt.title("Annotated Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(bounding_box_image)
    plt.title("Bounding Box Image")
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(30, 12))
    plt.imshow(detected_contour_image)
    plt.title("Detected Contour Image")
    plt.axis('off')
    plt.show()
    
    cv2.imwrite(output_file_suff+"_annotated.jpg", annotated_image)
    cv2.imwrite(output_file_suff+"_bounding_box.jpg", bounding_box_image)
    cv2.imwrite(output_file_suff+"_detected_contours.jpg", detected_contour_image)
    print("Images Saved Successfully")
    
    return cell_data


def analyze_and_label_objects_sci(image, output="excel", excel_path="sci_Cell_Data.xlsx"):
    if output not in ["excel", "console"]:
        print("Enter Valid Mode for function")
        return -1
    # Load image and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Label image regions
    labeled_img = label(binary)

    # Data storage
    cell_data = []
    
    # Ensure you provide the intensity image when calling regionprops
    regions = regionprops(labeled_img, intensity_image=gray)  # gray_img should be the corresponding grayscale image

    for i, region in enumerate(regions):
        # Get contour for additional perimeter calculation
        contour = contours[i] if i < len(contours) else None
        perimeter = cv2.arcLength(contour, True) if contour is not None else 0

        # Compute circularity
        circularity = (4 * np.pi * region.area) / (perimeter ** 2) if perimeter > 0 else 0

        # Compute convex hull properties
        convex_hull = convex_hull_image(region.image)
        convex_area = np.sum(convex_hull)

        # Store data
        cell_data.append({
            "Label": i + 1,
            "Area": region.area,
            "Bounding Box": region.bbox,
            "Centroid X": region.centroid[1],
            "Centroid Y": region.centroid[0],
            "Perimeter": region.perimeter,
            "Circularity": circularity,
            "Eccentricity": region.eccentricity,
            "EquivDiameter": region.equivalent_diameter,
            "Extent": region.extent,
            "Solidity": region.solidity,
            "Major Axis Length": region.major_axis_length,
            "Minor Axis Length": region.minor_axis_length,
            "Orientation": region.orientation,
            "Convex Area": convex_area,
            "Mean Intensity": region.mean_intensity,  # Now valid because intensity_image is provided
            "Aspect Ratio": region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
        })


    # Convert to DataFrame
    df = pd.DataFrame(cell_data)

    # Output handling
    if output == "excel":
        df.to_excel(excel_path, index=False)
        print(f"Data saved to {excel_path}")
    else:
        print(df)


def main():

    image_path = sys.argv[1]
    option=sys.argv[2]
    neighborhood_size=sys.argv[3]
    print("Selected Options For Analysis:", option,"\n*******************")

    
    out_dir="Output/Results/"
    image_path="/home/surajit/CV/Unsupervised_Learning_NEW/341.jpg"
    option="Geometry,General,Voronoi Entropy"
    neighborhood_size=7

    
    if len(sys.argv) != 4:
       print("Usage: python imaproc.py <excel_file>", sys.argv[1])
       sys.exit(1)

    image_path = sys.argv[1]
    option=sys.argv[2]
    neighborhood_size=sys.argv[3]
    print("Selected Options For Analysis:", option,"\n*******************")


    

    
    
        
        
    # Extract file name and extension
    
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    # Create output file name
    #input_img = f"out_{file_name}.jpg"
    input_img = out_dir+f"0_{file_name}_{neighborhood_size}{file_extension}"
    output_file_suff = out_dir+f"0_{file_name}_{neighborhood_size}"
    analyze_and_label_objects(input_img, out_dir, output_file_suff, option, file_name)
    
    #image_original = cv2.imread(image_path)
    #analyze_and_label_objects(image_original)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e)
    finally:
        #input("\nPress Enter to exit...")
        pass
    #cProfile.run('main()') #main()
