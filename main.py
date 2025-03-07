import cv2
import numpy as np
import os
import argparse
from typing import List, Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StampExtractor:
    """
    A class to extract stamps from document images.
    Supports various types of stamps including colored stamps and circular stamps.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the StampExtractor with configuration options.
        
        Args:
            config (Dict, optional): Configuration parameters for stamp extraction.
        """
        # Default configuration
        self.config = {
            'red_stamp': {
                'lower_hsv': np.array([0, 120, 70]),
                'upper_hsv': np.array([10, 255, 255]),
                'second_lower_hsv': np.array([170, 120, 70]),
                'second_upper_hsv': np.array([180, 255, 255])
            },
            'blue_stamp': {
                'lower_hsv': np.array([100, 100, 70]),
                'upper_hsv': np.array([140, 255, 255])
            },
            'circular_stamp': {
                'min_radius': 50,
                'max_radius': 300,
                'param1': 30,
                'param2': 60,
                'min_dist': 80
            },
            'general_stamp': {
                'min_area': 5000,
                'max_area': 100000,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0
            }
        }
        
        # Update config with user-provided values
        if config:
            self._update_config(config)
            
        logger.info("StampExtractor initialized with configuration")
    
    def _update_config(self, config: Dict) -> None:
        """
        Update the configuration with user-provided values.
        
        Args:
            config (Dict): User-provided configuration parameters.
        """
        for category, params in config.items():
            if category in self.config:
                self.config[category].update(params)
            else:
                self.config[category] = params
    
    def extract_stamps_from_file(self, file_path: str, output_dir: str = './extracted_stamps', 
                               extract_red: bool = True, extract_blue: bool = False, 
                               extract_circular: bool = True, extract_general: bool = True) -> List[str]:
        """
        Extract stamps from an image file and save them to the output directory.
        
        Args:
            file_path (str): Path to the input image file.
            output_dir (str, optional): Directory to save extracted stamps.
            extract_red (bool, optional): Whether to extract red stamps.
            extract_blue (bool, optional): Whether to extract blue stamps.
            extract_circular (bool, optional): Whether to extract circular stamps.
            extract_general (bool, optional): Whether to extract general stamps.
            
        Returns:
            List[str]: Paths to the extracted stamp images.
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the image
        try:
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Failed to read image: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error reading image {file_path}: {str(e)}")
            return []
        
        # Extract stamps
        extracted_paths = []
        
        # Extract red stamps if requested
        if extract_red:
            red_stamps = self._extract_colored_stamps(image, 'red_stamp')
            for i, stamp in enumerate(red_stamps):
                output_path = os.path.join(output_dir, f"red_stamp_{i}_{os.path.basename(file_path)}")
                cv2.imwrite(output_path, stamp)
                extracted_paths.append(output_path)
                logger.info(f"Extracted red stamp {i} to {output_path}")
                
        # Extract blue stamps if requested
        if extract_blue:
            blue_stamps = self._extract_colored_stamps(image, 'blue_stamp')
            for i, stamp in enumerate(blue_stamps):
                output_path = os.path.join(output_dir, f"blue_stamp_{i}_{os.path.basename(file_path)}")
                cv2.imwrite(output_path, stamp)
                extracted_paths.append(output_path)
                logger.info(f"Extracted blue stamp {i} to {output_path}")
                
        # Extract circular stamps if requested
        if extract_circular:
            circular_stamps = self._extract_circular_stamps(image)
            for i, stamp in enumerate(circular_stamps):
                output_path = os.path.join(output_dir, f"circular_stamp_{i}_{os.path.basename(file_path)}")
                cv2.imwrite(output_path, stamp)
                extracted_paths.append(output_path)
                logger.info(f"Extracted circular stamp {i} to {output_path}")
                
        # Extract general stamps if requested
        if extract_general:
            general_stamps = self._extract_general_stamps(image)
            for i, stamp in enumerate(general_stamps):
                output_path = os.path.join(output_dir, f"general_stamp_{i}_{os.path.basename(file_path)}")
                cv2.imwrite(output_path, stamp)
                extracted_paths.append(output_path)
                logger.info(f"Extracted general stamp {i} to {output_path}")
                
        return extracted_paths
    
    def _extract_colored_stamps(self, image: np.ndarray, color_type: str) -> List[np.ndarray]:
        """
        Extract stamps of a specific color from the image.
        
        Args:
            image (np.ndarray): Input image.
            color_type (str): Type of color to extract (e.g., 'red_stamp', 'blue_stamp').
            
        Returns:
            List[np.ndarray]: List of extracted stamp images.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get color configuration
        color_config = self.config.get(color_type, {})
        
        # Create mask for the specified color
        mask = None
        
        if color_type == 'red_stamp':
            # Red color is at both ends of the HSV spectrum, so we need two ranges
            lower_mask = cv2.inRange(hsv, color_config['lower_hsv'], color_config['upper_hsv'])
            upper_mask = cv2.inRange(hsv, color_config['second_lower_hsv'], color_config['second_upper_hsv'])
            mask = cv2.bitwise_or(lower_mask, upper_mask)
        else:
            mask = cv2.inRange(hsv, color_config['lower_hsv'], color_config['upper_hsv'])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stamps = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config['general_stamp']['min_area'] and area < self.config['general_stamp']['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h
                if (aspect_ratio >= self.config['general_stamp']['min_aspect_ratio'] and 
                    aspect_ratio <= self.config['general_stamp']['max_aspect_ratio']):
                    # Extract the region
                    stamp_image = image[y:y+h, x:x+w].copy()
                    
                    # Create a mask for the contour
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    contour_shifted = contour - np.array([x, y])
                    cv2.drawContours(contour_mask, [contour_shifted], 0, 255, -1)
                    
                    # Apply the mask to keep only the stamp pixels
                    for c in range(3):  # Apply mask to each color channel
                        stamp_image[:, :, c] = cv2.bitwise_and(stamp_image[:, :, c], 
                                                              stamp_image[:, :, c], 
                                                              mask=contour_mask)
                    
                    stamps.append(stamp_image)
        
        return stamps
    
    def _extract_circular_stamps(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract circular stamps from the image using Hough Circle Transform.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            List[np.ndarray]: List of extracted circular stamp images.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.config['circular_stamp']['min_dist'],
            param1=self.config['circular_stamp']['param1'],
            param2=self.config['circular_stamp']['param2'],
            minRadius=self.config['circular_stamp']['min_radius'],
            maxRadius=self.config['circular_stamp']['max_radius']
        )
        
        stamps = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Get the coordinates and radius
                x, y, radius = circle
                
                # Create a bounding box around the circle
                box_x = max(0, x - radius)
                box_y = max(0, y - radius)
                box_width = min(2 * radius, image.shape[1] - box_x)
                box_height = min(2 * radius, image.shape[0] - box_y)
                
                # Extract the region
                stamp_image = image[box_y:box_y+box_height, box_x:box_x+box_width].copy()
                
                # Create a circular mask
                mask = np.zeros((box_height, box_width), dtype=np.uint8)
                center_x, center_y = radius, radius
                if box_x == 0:
                    center_x = x
                if box_y == 0:
                    center_y = y
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # Apply the mask to keep only the stamp pixels
                for c in range(3):  # Apply mask to each color channel
                    stamp_image[:, :, c] = cv2.bitwise_and(stamp_image[:, :, c], 
                                                          stamp_image[:, :, c], 
                                                          mask=mask)
                
                stamps.append(stamp_image)
        
        return stamps
    
    def _extract_general_stamps(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract general stamps from the image using contour analysis.
        This method tries to detect stamps that are neither color-specific nor necessarily circular.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            List[np.ndarray]: List of extracted stamp images.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        morph_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract stamp regions
        stamps = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config['general_stamp']['min_area'] and area < self.config['general_stamp']['max_area']:
                # Get the bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h
                if (aspect_ratio >= self.config['general_stamp']['min_aspect_ratio'] and 
                    aspect_ratio <= self.config['general_stamp']['max_aspect_ratio']):
                    
                    # Extract the region
                    stamp_image = image[y:y+h, x:x+w].copy()
                    
                    # Create a mask for the contour
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    contour_shifted = contour - np.array([x, y])
                    cv2.drawContours(contour_mask, [contour_shifted], 0, 255, -1)
                    
                    # Apply the mask to keep only the stamp pixels
                    for c in range(3):  # Apply mask to each color channel
                        stamp_image[:, :, c] = cv2.bitwise_and(stamp_image[:, :, c], 
                                                              stamp_image[:, :, c], 
                                                              mask=contour_mask)
                    
                    stamps.append(stamp_image)
        
        return stamps
    
    def extract_stamps_from_dir(self, dir_path: str, output_dir: str = './extracted_stamps', 
                              file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']) -> Dict[str, List[str]]:
        """
        Extract stamps from all images in a directory.
        
        Args:
            dir_path (str): Path to the directory containing images.
            output_dir (str, optional): Directory to save extracted stamps.
            file_extensions (List[str], optional): File extensions to process.
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping input files to their extracted stamp paths.
        """
        if not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return {}
        
        results = {}
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in file_extensions):
                extracted_paths = self.extract_stamps_from_file(file_path, output_dir)
                results[file_path] = extracted_paths
                
        return results

# Main function for command-line usage
def main():
    parser = argparse.ArgumentParser(description='Extract stamps from document images.')
    parser.add_argument('-i', '--input', required=True, help='Input image file or directory')
    parser.add_argument('-o', '--output', default='./extracted_stamps', help='Output directory for extracted stamps')
    parser.add_argument('--red', action='store_true', help='Extract red stamps')
    parser.add_argument('--blue', action='store_true', help='Extract blue stamps')
    parser.add_argument('--circular', action='store_true', help='Extract circular stamps')
    parser.add_argument('--general', action='store_true', help='Extract general stamps')
    parser.add_argument('--all', action='store_true', help='Extract all types of stamps')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = StampExtractor()
    
    # Set stamp types to extract
    extract_red = args.red or args.all
    extract_blue = args.blue or args.all
    extract_circular = args.circular or args.all
    extract_general = args.general or args.all
    
    # If no specific type is selected, extract all
    if not (extract_red or extract_blue or extract_circular or extract_general):
        extract_red = extract_circular = extract_general = True
    
    # Process input
    if os.path.isfile(args.input):
        extracted_paths = extractor.extract_stamps_from_file(
            args.input, args.output, extract_red, extract_blue, extract_circular, extract_general
        )
        logger.info(f"Extracted {len(extracted_paths)} stamps from {args.input}")
    elif os.path.isdir(args.input):
        results = extractor.extract_stamps_from_dir(args.input, args.output)
        total_stamps = sum(len(paths) for paths in results.values())
        logger.info(f"Extracted {total_stamps} stamps from {len(results)} files in {args.input}")
    else:
        logger.error(f"Input not found: {args.input}")

if __name__ == "__main__":
    main()
