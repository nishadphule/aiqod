import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Utility and Processing Functions
# -----------------------------

def remove_black_text(image_bgr):
    """
    Removes black regions (e.g., text) from the image using LAB color space.
    Black text is identified as regions with low lightness and neutral color.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lower_black = np.array([0, 118, 118], dtype=np.uint8)
    upper_black = np.array([128, 138, 138], dtype=np.uint8)
    black_mask = cv2.inRange(lab, lower_black, upper_black)
    image_no_black = image_bgr.copy()
    image_no_black[black_mask > 0] = [255, 255, 255]
    return image_no_black

def is_stamp_shape(cnt):
    """Check if a contour is circular or rectangular."""
    area = cv2.contourArea(cnt)
    if area == 0:
        return False
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box_area = cv2.contourArea(box)
    if box_area == 0:
        return False
    rectangularity = area / box_area
    return circularity > 0.7 or rectangularity > 0.7

def detect_stamp(image_bgr):
    """
    Detects blue or red stamps using HSV color thresholding and shape constraints.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Blue range
    lower_blue = np.array([85, 80, 50])
    upper_blue = np.array([150, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Red range (two parts due to hue wrap-around)
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([160, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # Clean up masks
    kernel = np.ones((7, 7), np.uint8)
    blue_mask_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask_clean = cv2.morphologyEx(blue_mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    # Find contours and filter by shape
    blue_contours, _ = cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_stamps = [cnt for cnt in blue_contours if cv2.contourArea(cnt) > 500 and is_stamp_shape(cnt)]
    red_contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_stamps = [cnt for cnt in red_contours if cv2.contourArea(cnt) > 500 and is_stamp_shape(cnt)]
    stamps = [{'contour': cnt, 'color': 'blue'} for cnt in blue_stamps] + \
             [{'contour': cnt, 'color': 'red'} for cnt in red_stamps]
    return stamps

# Define target colors in RGB for signature and stamp extraction
SIG_TARGET_RGB = np.array([[[175, 169, 198]]], dtype=np.uint8)  # Muted purple for signature
STAMP_TARGET_RGB = np.array([[[114, 155, 225]]], dtype=np.uint8)  # Light blue for stamp
SIG_TARGET_LAB = cv2.cvtColor(SIG_TARGET_RGB, cv2.COLOR_RGB2LAB)[0, 0]
STAMP_TARGET_LAB = cv2.cvtColor(STAMP_TARGET_RGB, cv2.COLOR_RGB2LAB)[0, 0]

def extract_signature_from_stamp(cropped_stamp_bgr):
    """
    Extracts the signature from a cropped stamp region by comparing color distances,
    using Otsu thresholding, and performing morphological cleanup.
    Returns a binary mask of the signature.
    """
    lab = cv2.cvtColor(cropped_stamp_bgr, cv2.COLOR_BGR2LAB)
    sig_dist_sq = np.sum((lab - SIG_TARGET_LAB) ** 2, axis=2)
    stamp_dist_sq = np.sum((lab - STAMP_TARGET_LAB) ** 2, axis=2)
    color_threshold = 10000  # Squared distance threshold
    closer_to_sig = (sig_dist_sq < stamp_dist_sq)
    within_sig_dist = (sig_dist_sq < color_threshold)
    sig_mask_color = np.where(closer_to_sig & within_sig_dist, 255, 0).astype(np.uint8)
    hsv = cv2.cvtColor(cropped_stamp_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    _, otsu_mask = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sig_mask = cv2.bitwise_and(sig_mask_color, otsu_mask)
    kernel = np.ones((3, 3), np.uint8)
    sig_mask = cv2.morphologyEx(sig_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    sig_mask = cv2.morphologyEx(sig_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sig_mask, connectivity=8)
    min_area = 200  # Exclude small components
    clean_mask = np.zeros_like(sig_mask)
    for i in range(1, num_labels):
        area_i = stats[i, cv2.CC_STAT_AREA]
        if area_i >= min_area:
            component_mask = np.zeros_like(labels, dtype=np.uint8)
            component_mask[labels == i] = 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = contours[0]
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            if box_area == 0:
                continue
            rectangularity = area / box_area
            if not (circularity > 0.7 or rectangularity > 0.7):
                clean_mask[labels == i] = 255
    return clean_mask

def segment_stamp_and_signature(image_bgr):
    """
    Segments stamps and signatures from an input image (provided as a BGR numpy array).
    Handles overlaps and returns a dictionary containing various visualization images and stats.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_no_black = remove_black_text(image_bgr)
    stamps = detect_stamp(image_no_black)
    stamp_mask = np.zeros(image_no_black.shape[:2], dtype=np.uint8)
    for stamp in stamps:
        cv2.drawContours(stamp_mask, [stamp['contour']], -1, 255, -1)
    overlapping_stamps = []
    signature_masks_per_stamp = []
    for stamp in stamps:
        cnt = stamp['contour']
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_stamp_bgr = image_no_black[y:y+h, x:x+w]
        signature_mask_crop = extract_signature_from_stamp(cropped_stamp_bgr)
        stamp_individual_mask = np.zeros_like(stamp_mask)
        cv2.drawContours(stamp_individual_mask, [cnt], -1, 255, -1)
        stamp_mask_crop = stamp_individual_mask[y:y+h, x:x+w]
        overlap_mask = cv2.bitwise_and(stamp_mask_crop, signature_mask_crop)
        if np.any(overlap_mask > 0):
            cropped_stamp_region = cv2.cvtColor(cropped_stamp_bgr, cv2.COLOR_BGR2RGB)
            stamp_crop = np.ones_like(cropped_stamp_region) * 255
            stamp_crop[stamp_mask_crop > 0] = cropped_stamp_region[stamp_mask_crop > 0]
            stamp_crop[signature_mask_crop > 0] = [255, 255, 255]
            overlapping_stamps.append(stamp_crop)
        full_signature_mask = np.zeros_like(stamp_mask)
        full_signature_mask[y:y+h, x:x+w][signature_mask_crop > 0] = 255
        signature_masks_per_stamp.append(full_signature_mask)
    signature_mask = np.zeros_like(stamp_mask)
    for mask in signature_masks_per_stamp:
        signature_mask = cv2.bitwise_or(signature_mask, mask)
    stamp_binary = stamp_mask > 0
    sig_binary = signature_mask > 0
    overlap = np.logical_and(stamp_binary, sig_binary)
    has_overlap = np.any(overlap)
    stamp_area = np.sum(stamp_binary)
    overlap_area = np.sum(overlap)
    overlap_percentage = (overlap_area / stamp_area * 100) if stamp_area > 0 else 0
    white_bg = np.ones_like(image_rgb) * 255
    stamp_only = white_bg.copy()
    stamp_only[stamp_binary] = image_rgb[stamp_binary]
    signature_only = white_bg.copy()
    signature_only[sig_binary] = image_rgb[sig_binary]
    signature_crop = None
    if np.any(signature_mask):
        rows, cols = np.where(signature_mask > 0)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        x_crop, y_crop = min_col, min_row
        w_crop, h_crop = max_col - min_col + 1, max_row - min_row + 1
        cropped_region = image_rgb[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        sig_mask_crop = signature_mask[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        signature_crop = np.ones_like(cropped_region) * 255
        signature_crop[sig_mask_crop > 0] = cropped_region[sig_mask_crop > 0]
    stats = {
        "total_stamps": len(stamps),
        "image_resolution": f"{image_rgb.shape[1]}x{image_rgb.shape[0]}"
    }
    return {
        'original': image_rgb,
        'image_no_black': cv2.cvtColor(image_no_black, cv2.COLOR_BGR2RGB),
        'stamp': stamp_only,
        'signature': signature_only,
        'stamp_mask': stamp_mask,
        'signature_mask': signature_mask,
        'has_overlap': has_overlap,
        'overlap_percentage': overlap_percentage,
        'overlapping_stamps': overlapping_stamps,
        'signature_crop': signature_crop,
        'stats': stats
    }

# -----------------------------
# Display Functions (Streamlit)
# -----------------------------

def display_results_streamlit(result):
    """
    Creates matplotlib figures to show the original image,
    processed images, overlap visualization, and any overlapping stamps.
    The figures are rendered in Streamlit.
    """
    orig_rgb = result['original']
    stamp_binary = result['stamp_mask'] > 0
    sig_binary = result['signature_mask'] > 0
    overlap = np.logical_and(stamp_binary, sig_binary)
    overlap_viz = orig_rgb.copy()
    overlap_viz[overlap] = [255, 0, 0]  # Highlight overlap in red
    
    # First figure: Main segmentation results
    fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("ORIGINAL IMAGE", fontsize=15)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(result['image_no_black'])
    axes[0, 1].set_title("BLACK TEXT REMOVED", fontsize=15)
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(result['stamp'])
    axes[0, 2].set_title("STAMP ONLY", fontsize=15)
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(result['signature'])
    axes[1, 0].set_title("SIGNATURE ONLY", fontsize=15)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(overlap_viz)
    axes[1, 1].set_title("OVERLAP VISUALIZATION", fontsize=15)
    axes[1, 1].axis("off")
    
    axes[1, 2].axis("off")
    overlap_text = (
        f"OVERLAP DETECTED: {result['overlap_percentage']:.2f}%"
        if result['has_overlap'] else "NO OVERLAP DETECTED"
    )
    fig1.text(0.5, 0.01, overlap_text, ha="center", fontsize=14,
              bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    fig1.tight_layout()
    st.pyplot(fig1)
    
    # Second figure: Overlapping stamps and signature crop (if any)
    if result['overlapping_stamps']:
        st.write(f"Overlapping stamps detected: {len(result['overlapping_stamps'])} stamp(s) overlap with the signature!")
        num_stamps = len(result['overlapping_stamps'])
        fig2, axes = plt.subplots(1, num_stamps + 1, figsize=(5 * (num_stamps + 1), 5))
        if result['signature_crop'] is not None:
            axes[0].imshow(result['signature_crop'])
            axes[0].set_title("Signature", fontsize=12)
            axes[0].axis("off")
        else:
            axes[0].text(0.5, 0.5, "No signature detected", ha="center", va="center")
            axes[0].axis("off")
        for i, stamp in enumerate(result['overlapping_stamps']):
            axes[i + 1].imshow(stamp)
            axes[i + 1].set_title(f"Overlapping Stamp {i + 1}", fontsize=12)
            axes[i + 1].axis("off")
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.write("No overlaps between stamps and signature.")

# -----------------------------
# Main Streamlit App
# -----------------------------

def main():
    st.title("Stamp and Sign Detector")
    st.write("Upload an image to detect stamps and signatures.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the image using PIL and convert to RGB
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        # Convert RGB to BGR for OpenCV processing
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            result = segment_stamp_and_signature(image_bgr)
        
        st.subheader("Detection Results")
        display_results_streamlit(result)
        
        st.subheader("Statistics")
        for key, value in result['stats'].items():
            st.write(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
