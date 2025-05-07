import cv2
import gc
import glob
import os
from PIL import Image
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from collections import Counter


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_data_paths(base_path):
    """Get data paths with caching directory"""
    original_dir = f"{base_path}/patch"
    binary_dir = f"{base_path}/patch-binary"
    color_dir = f"{base_path}/patch-texture"
    cache_dir = f"{base_path}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    return original_dir, binary_dir, color_dir, cache_dir


def get_device():
    """Get available device and set optimal settings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # Optimize GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ...existing code...

def load_tiled_data_fast(original_dir, binary_dir, color_dir, cache_dir):
    """Fast data loading with caching and parallel processing"""
    cache_file = os.path.join(cache_dir, 'tiled_data_paths.pkl')

    # Try to load from cache
    if os.path.exists(cache_file):
        print("📁 Loading paths from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("🔍 Scanning directories for matching files...")
    filenames = sorted(os.listdir(original_dir))

    def check_file_exists(filename):
        """Check if all three corresponding files exist"""
        img_path = os.path.join(original_dir, filename)
        binary_path = os.path.join(binary_dir, filename)
        color_path = os.path.join(color_dir, filename)

        if os.path.exists(img_path) and os.path.exists(binary_path) and os.path.exists(color_path):
            return img_path, binary_path, color_path
        return None

    # Use parallel processing to check file existence
    with ThreadPoolExecutor(max_workers=min(32, cpu_count())) as executor:
        results = list(tqdm(
            executor.map(check_file_exists, filenames),
            total=len(filenames),
            desc="Checking files"
        ))

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    image_paths, binary_paths, color_paths = zip(*valid_results) if valid_results else ([], [], [])

    # Convert to lists
    image_paths, binary_paths, color_paths = list(image_paths), list(binary_paths), list(color_paths)

    # Cache the results
    with open(cache_file, 'wb') as f:
        pickle.dump((image_paths, binary_paths, color_paths), f)

    print(f"💾 Cached {len(image_paths)} file paths")
    return image_paths, binary_paths, color_paths


def process_single_tile_gpu(args):
    """Process a single tile with GPU acceleration where possible"""
    idx, image_path, binary_path, color_path, min_contour_area, target_size, debug_mode = args

    try:
        # Load images with optimized settings
        with Image.open(image_path) as img:
            orig_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        with Image.open(binary_path) as img:
            binary_mask = np.array(img.convert("L"), dtype=np.float32) / 255.0

        with Image.open(color_path) as img:
            color_mask = np.array(img.convert("RGB"), dtype=np.uint8)

        # Convert binary mask for contour detection
        binary_cv = (binary_mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        patches = []
        labels = []
        tile_info = []

        if valid_contours:
            # Sort contours by area (largest first)
            valid_contours.sort(key=cv2.contourArea, reverse=True)

            for contour_idx, contour in enumerate(valid_contours):
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(orig_img.shape[1] - x, w + 2 * padding)
                h = min(orig_img.shape[0] - y, h + 2 * padding)

                # Extract patches
                orig_patch = orig_img[y:y + h, x:x + w]
                color_patch = color_mask[y:y + h, x:x + w]

                # Skip if patch is too small
                if orig_patch.shape[0] < 20 or orig_patch.shape[1] < 20:
                    continue

                # Resize patches using optimized OpenCV
                orig_patch_resized = cv2.resize(orig_patch, target_size, interpolation=cv2.INTER_AREA)
                color_patch_resized = cv2.resize(color_patch, target_size, interpolation=cv2.INTER_AREA)

                # Get label from color patch
                label, pixel_counts = get_roof_texture_label_optimized(
                    color_patch_resized.astype(np.uint8),
                    min_pixel_threshold=50,
                    tolerance=15
                )

                # Calculate confidence
                total_colored = sum([pixel_counts["smooth"], pixel_counts["average"], pixel_counts["rough"]])
                if total_colored > 0:
                    max_colored = max(pixel_counts["smooth"], pixel_counts["average"], pixel_counts["rough"])
                    confidence = max_colored / total_colored
                else:
                    confidence = 0

                patches.append(orig_patch_resized)
                labels.append(label)
                tile_info.append({
                    'tile_idx': idx,
                    'contour_idx': contour_idx,
                    'contour_area': cv2.contourArea(contour),
                    'bbox': (x, y, w, h),
                    'pixel_counts': pixel_counts,
                    'confidence': confidence,
                    'classification_ratio': total_colored / (target_size[0] * target_size[1])
                })

        else:
            # Use full image
            orig_patch = cv2.resize(orig_img, target_size, interpolation=cv2.INTER_AREA)
            color_patch_full = cv2.resize(color_mask, target_size, interpolation=cv2.INTER_AREA)

            label, pixel_counts = get_roof_texture_label_optimized(
                color_patch_full.astype(np.uint8),
                min_pixel_threshold=30
            )

            patches.append(orig_patch)
            labels.append(label)
            tile_info.append({
                'tile_idx': idx,
                'contour_idx': -1,
                'contour_area': 0,
                'bbox': (0, 0, orig_img.shape[1], orig_img.shape[0]),
                'pixel_counts': pixel_counts,
                'confidence': 0,
                'classification_ratio': 0
            })

        return patches, labels, tile_info

    except Exception as e:
        print(f"Error processing tile {idx}: {str(e)}")
        return [], [], []


def get_roof_texture_label_optimized(color_patch, min_pixel_threshold=50, tolerance=10):
    """Optimized version using vectorized operations"""
    # Define class colors as numpy arrays
    class_colors = {
        "smooth": np.array([83, 217, 56], dtype=np.uint8),
        "average": np.array([252, 240, 3], dtype=np.uint8),
        "rough": np.array([255, 0, 0], dtype=np.uint8)
    }

    # Flatten image for vectorized comparison
    pixels = color_patch.reshape(-1, 3)

    # Vectorized color matching
    pixel_counts = {}
    for class_name, target_color in class_colors.items():
        # Use broadcasting for efficient distance calculation
        distances = np.abs(pixels - target_color[None, :]).max(axis=1)
        pixel_counts[class_name] = np.sum(distances <= tolerance)

    # Count background pixels
    pixel_counts["background"] = np.sum(np.all(pixels <= 10, axis=1))

    # Determine label
    class_counts = {k: pixel_counts[k] for k in ["smooth", "average", "rough"]}
    label, max_count = max(class_counts.items(), key=lambda x: x[1])

    if max_count < min_pixel_threshold:
        return "no_contour", pixel_counts
    else:
        return label, pixel_counts


def extract_multiple_rooftops_per_tile_fast(image_paths, binary_paths, color_paths,
                                            min_contour_area=100, target_size=(224, 224),
                                            debug_mode=False, cache_dir=None, batch_size=50):
    """Fast parallel extraction with caching and batching"""

    cache_file = None
    if cache_dir:
        cache_file = os.path.join(cache_dir,
                                  f'extracted_patches_{len(image_paths)}_{min_contour_area}_{target_size[0]}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            print("📁 Loading extracted patches from cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                print(f"✅ Loaded {len(cached_data[0])} cached patches")
                return cached_data

    print("🚀 Starting fast parallel patch extraction...")

    # Prepare arguments for parallel processing
    args_list = [
        (idx, image_paths[idx], binary_paths[idx], color_paths[idx],
         min_contour_area, target_size, debug_mode)
        for idx in range(len(image_paths))
    ]

    all_patches = []
    all_labels = []
    all_tile_info = []

    # Process in batches to manage memory
    num_workers = min(cpu_count(), 16)  # Limit workers to prevent memory issues

    for batch_start in tqdm(range(0, len(args_list), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(args_list))
        batch_args = args_list[batch_start:batch_end]

        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            batch_results = list(tqdm(
                executor.map(process_single_tile_gpu, batch_args),
                total=len(batch_args),
                desc=f"Batch {batch_start // batch_size + 1}",
                leave=False
            ))

        # Collect results from this batch
        for patches, labels, tile_info in batch_results:
            all_patches.extend(patches)
            all_labels.extend(labels)
            all_tile_info.extend(tile_info)

        # Force garbage collection after each batch
        gc.collect()

    # Convert to numpy array
    patches_array = np.array(all_patches) if all_patches else np.array([])

    print(f"✅ Extracted {len(all_patches)} patches total")

    # Cache the results
    if cache_file:
        print("💾 Caching extracted patches...")
        with open(cache_file, 'wb') as f:
            pickle.dump((patches_array, all_labels, all_tile_info), f)
        print(f"✅ Cached results to {cache_file}")

    if debug_mode and all_labels:
        print("\n=== EXTRACTION SUMMARY ===")
        label_counts = Counter(all_labels)
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    return patches_array, all_labels, all_tile_info


# ...existing code...

def main_fast():
    """Optimized main function with caching and GPU acceleration"""
    set_seed(42)
    device = get_device()

    base_path = "/home/student/sky-scan/data"
    original_dir, binary_dir, color_dir, cache_dir = get_data_paths(base_path)

    # Fast data loading with caching
    image_paths, binary_paths, color_paths = load_tiled_data_fast(
        original_dir, binary_dir, color_dir, cache_dir
    )
    print("len of images path :", len(image_paths))
    print("len of binary images path :", len(binary_paths))
    print("len of colour images path :", len(color_paths))

    # Fast extraction with parallel processing and caching
    patches, labels, tile_info = extract_multiple_rooftops_per_tile_fast(
        image_paths, binary_paths, color_paths,
        min_contour_area=100,
        target_size=(224, 224),
        debug_mode=True,
        cache_dir=cache_dir,
        batch_size=100  # Increase batch size for faster GPUs
    )

    return patches, labels, tile_info


# Run the optimized version
patches, labels, tile_info = main_fast()
print("len of patches:", len(patches))
print("🚀 Using optimized fast extraction - see previous cells")
# generate model here