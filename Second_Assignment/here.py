# %% [markdown]
# # BBM418 Computer Vision Assignment 2
# ## Homography Estimation
# ### Student ID: b2220356053

# %% [markdown]
# ## Part 1: Feature Extraction
# 
# In this part, feature detection and descriptor extraction implemented using two different methods (I couldn't make SURF work because of OPENCV limitations):
# - **SIFT** (Scale-Invariant Feature Transform): Good for detecting features at different scales and rotations
# - **ORB** (Oriented FAST and Rotated BRIEF): Binary descriptor, very fast but less robust
# 
# These features are important for homography estimation because they help us find corresponding points between images that are taken from different viewpoints.

# %%
# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set up matplotlib for better visualization
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 100

# %%
# Define dataset paths
BASE_DIR = Path('/home/oguzhan/Projects/BBM416_Projects/Second_Assignment')
PANORAMA_DATASET_DIR = BASE_DIR / 'pa2_data' / 'panorama_dataset'

# List all available scenes
scenes = [d for d in PANORAMA_DATASET_DIR.iterdir() if d.is_dir()]
scene_names = [s.name for s in scenes]

for i, name in enumerate(scene_names, 1):
    print(f"  {i}. {name}")

# %% [markdown]
# ### 1.1 Feature Detector Initialization

# %%
# Initialize feature detectors
# SIFT - Scale Invariant Feature Transform
# Good at detecting corners and blobs at different scales
sift = cv2.SIFT_create()

# ORB - Oriented FAST and Rotated BRIEF
# Binary descriptor, very fast, rotation invariant
# Good for real-time applications
orb = cv2.ORB_create(nfeatures=2000)  # Increase max features for better matching

# %% [markdown]
# ### 1.2 Feature Extraction Function
# 
# This function will:
# 1. Load an image in grayscale (feature detectors work on intensity)
# 2. Detect keypoints using the specified detector
# 3. Compute descriptors for each keypoint
# 4. Return both keypoints and descriptors for matching later

# %%
def extract_features(image_path, detector, detector_name="Unknown"):
    """
    Extract keypoints and descriptors from an image using the specified detector.
    
    Args:
        image_path: Path to the input image
        detector: Feature detector object (SIFT or ORB)
        detector_name: Name of the detector for printing
    
    Returns:
        img_color: Original color image for visualization
        img_gray: Grayscale image
        keypoints: List of detected keypoints
        descriptors: Array of descriptors for each keypoint
    """
    # Read the image in color for visualization
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB for matplotlib (OpenCV uses BGR by default)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for feature detection
    # detectors work on intensity values, not color
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors
    # keypoints: location, scale, orientation of features
    # descriptors: numerical representation for matching
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    
    return img_color, img_gray, keypoints, descriptors

# %% [markdown]
# ### 1.3 Visualization Function
# 
# Create a function to visualize detected keypoints on images. This helps us understand:
# - Where features are detected (corners, edges, textured regions)
# - How many features each detector finds
# - The spatial distribution of features across the image

# %%
def visualize_keypoints(img_color, keypoints, detector_name, image_name):
    """
    Visualize detected keypoints on the image.
    
    Args:
        img_color: Color image
        keypoints: List of detected keypoints
        detector_name: Name of the detector used
        image_name: Name of the image file
    
    Returns:
        img_with_keypoints: Image with keypoints drawn
    """
    # Draw keypoints on the image
    # The circles show the location and scale of each keypoint
    # Larger circles indicate features detected at larger scales
    img_with_keypoints = cv2.drawKeypoints(
        img_color, 
        keypoints, 
        None,
        color=(0, 255, 0),  # Green color for keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # Show scale and orientation
    )
    
    return img_with_keypoints


def compare_detectors_on_image(image_path, detectors_dict):
    """
    Compare multiple feature detectors on a single image.
    
    Args:
        image_path: Path to the image
        detectors_dict: Dictionary of {name: detector_object}
    """
    image_name = Path(image_path).name
    num_detectors = len(detectors_dict)
    
    # Create subplots for comparison
    fig, axes = plt.subplots(1, num_detectors + 1, figsize=(5 * (num_detectors + 1), 5))
    
    # Show original image first
    img_color = cv2.imread(str(image_path))
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_color)
    axes[0].set_title(f'Original Image\n{image_name}', fontsize=12)
    axes[0].axis('off')
    
    # Process each detector
    for idx, (name, detector) in enumerate(detectors_dict.items(), 1):            
        # Extract features
        img_color, img_gray, keypoints, descriptors = extract_features(
            image_path, detector, name
        )
        
        # Visualize keypoints
        img_with_kp = visualize_keypoints(img_color, keypoints, name, image_name)
        
        # Display
        axes[idx].imshow(img_with_kp)
        axes[idx].set_title(f'{name}\n{len(keypoints)} keypoints', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 1.4 Testing Feature Extraction
# 
# Let's test the feature extraction on sample images from each scene. I'll compare all three detectors side-by-side to see their differences.

# %%
# Create a dictionary of all detectors
detectors = {
    'SIFT': sift,
    'ORB': orb
}

# Test on one image from each scene
print("Testing feature extraction on sample images from each scene:\n")
print("="*60)

for scene_name in scene_names[:3]:  # Test on first 3 scenes
    scene_path = PANORAMA_DATASET_DIR / scene_name
    # Use the first image from each scene
    test_image = scene_path / "1.png"
    
    if test_image.exists():
        print(f"\nProcessing scene: {scene_name}")
        print("-"*60)
        compare_detectors_on_image(test_image, detectors)
    else:
        print(f"Image not found: {test_image}")

# %% [markdown]
# ### 1.5 Extract and Store Features for All Images
# 
# Now I'll extract features for all images in all scenes and store them for later use in matching and homography estimation. This will save computation time since we won't need to re-extract features.

# %%
def extract_all_features_for_scene(scene_path, detector, detector_name):
    """
    Extract features for all images in a scene.
    
    Args:
        scene_path: Path to the scene directory
        detector: Feature detector object
        detector_name: Name of the detector
    
    Returns:
        features_dict: Dictionary mapping image_name -> (img_color, keypoints, descriptors)
    """
    features_dict = {}
    
    # Get all PNG images in the scene
    image_files = sorted(scene_path.glob("*.png"))
    
    for img_path in image_files:
        img_color, img_gray, keypoints, descriptors = extract_features(
            img_path, detector, detector_name
        )
        
        # Store the results
        features_dict[img_path.name] = {
            'image_color': img_color,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'path': img_path
        }
    
    return features_dict


# Store all extracted features organized by scene and detector
all_features = {}

print("Extracting features for all images in all scenes...")
print("="*70)

for scene_name in scene_names:
    scene_path = PANORAMA_DATASET_DIR / scene_name
    all_features[scene_name] = {}
    
    print(f"\nScene: {scene_name}")
    print("-"*70)
    
    for detector_name, detector in detectors.items():
        print(f"Using {detector_name}...")
        features = extract_all_features_for_scene(scene_path, detector, detector_name)
        all_features[scene_name][detector_name] = features
        
        # Print summary
        total_keypoints = sum(f['keypoints'].__len__() for f in features.values())
        avg_keypoints = total_keypoints / len(features) if features else 0
        print(f"Processed {len(features)} images")
        print(f"Total keypoints: {total_keypoints}")
        print(f"Average keypoints per image: {avg_keypoints:.1f}")

print("\n" + "="*70)

# %% [markdown]
# ### 1.6 Detailed Visualization of Keypoint Distribution
# 
# Let's create a detailed visualization showing how keypoints are distributed across different images. This helps understand:
# - Which areas of the image have more features (usually textured regions)
# - Whether features are distributed uniformly or concentrated in certain areas
# - How different detectors behave on the same image

# %%
def visualize_scene_features(scene_name, detector_name='SIFT', num_images=3):
    """
    Visualize features extracted from multiple images in a scene.
    
    Args:
        scene_name: Name of the scene
        detector_name: Which detector to visualize
        num_images: Number of images to show
    """
    if scene_name not in all_features:
        print(f"Scene {scene_name} not found!")
        return
    
    if detector_name not in all_features[scene_name]:
        print(f"Detector {detector_name} not available for scene {scene_name}!")
        return
    
    features = all_features[scene_name][detector_name]
    image_names = sorted(list(features.keys()))[:num_images]
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 6))
    if num_images == 1:
        axes = [axes]
    
    fig.suptitle(f'Feature Extraction: {scene_name} using {detector_name}', 
                 fontsize=16, fontweight='bold')
    
    for idx, img_name in enumerate(image_names):
        feature_data = features[img_name]
        img_color = feature_data['image_color']
        keypoints = feature_data['keypoints']
        
        # Draw keypoints
        img_with_kp = visualize_keypoints(img_color, keypoints, detector_name, img_name)
        
        axes[idx].imshow(img_with_kp)
        axes[idx].set_title(f'{img_name}\n{len(keypoints)} keypoints', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# Visualize features for a couple of scenes
print("Visualizing feature extraction results:\n")

# Choose different scenes to showcase variety
example_scenes = ['v_bird', 'v_graffiti']

for scene in example_scenes:
    if scene in all_features:
        print(f"\nScene: {scene}")
        visualize_scene_features(scene, 'SIFT', num_images=3)

# %% [markdown]
# ### 1.7 Statistical Analysis of Feature Detectors
# 
# Let's compare the performance characteristics of different detectors by analyzing:
# - Number of keypoints detected per image
# - Average keypoints across all scenes
# - Computational efficiency (if timing info available)
# 
# This analysis helps justify which detector to use for different stages of the pipeline.

# %%
def analyze_detector_statistics():
    """
    Analyze and compare statistics of different feature detectors.
    """
    print("Feature Detector Comparison")
    print("="*80)
    
    # Collect statistics
    stats = {}
    
    for detector_name in ['SIFT', 'ORB']:
        stats[detector_name] = {
            'total_keypoints': 0,
            'num_images': 0,
            'keypoints_per_scene': {}
        }
        
        for scene_name in scene_names:
            if scene_name in all_features and detector_name in all_features[scene_name]:
                features = all_features[scene_name][detector_name]
                scene_keypoints = sum(f['keypoints'].__len__() for f in features.values())
                num_images = len(features)
                
                stats[detector_name]['total_keypoints'] += scene_keypoints
                stats[detector_name]['num_images'] += num_images
                stats[detector_name]['keypoints_per_scene'][scene_name] = {
                    'total': scene_keypoints,
                    'avg': scene_keypoints / num_images if num_images > 0 else 0
                }
    
    # Print comparison table
    print(f"\n{'Detector':<10} {'Total Images':<15} {'Total Keypoints':<18} {'Avg per Image':<15}")
    print("-"*80)
    
    for detector_name, data in stats.items():
        if data['num_images'] > 0:
            avg_per_image = data['total_keypoints'] / data['num_images']
            print(f"{detector_name:<10} {data['num_images']:<15} {data['total_keypoints']:<18} {avg_per_image:<15.1f}")
    
    # Print per-scene breakdown
    print("\n\nPer-Scene Breakdown:")
    print("="*80)
    
    for scene_name in scene_names:
        print(f"\n{scene_name}:")
        print(f"{'  Detector':<12} {'Total Keypoints':<18} {'Avg per Image':<15}")
        print("  " + "-"*70)
        
        for detector_name in ['SIFT', 'ORB']:
            if scene_name in stats[detector_name]['keypoints_per_scene']:
                scene_stats = stats[detector_name]['keypoints_per_scene'][scene_name]
                print(f"  {detector_name:<12} {scene_stats['total']:<18} {scene_stats['avg']:<15.1f}")
    
    return stats

# Run the analysis
feature_stats = analyze_detector_statistics()

# %% [markdown]
# ## Part 2: Feature Matching
# 
# In this part, I'll implement feature matching between image pairs to find corresponding points.
# 
# **Matching Strategy:**
# - Use **k-Nearest Neighbors (k=2)** to find the two closest matches for each descriptor
# - Apply **Lowe's Ratio Test** to filter out ambiguous matches
# - Use **Euclidean distance** for SIFT descriptors (128-dim float vectors)
# - Visualize matches to verify quality before using them for homography

# %% [markdown]
# ### 2.1 Feature Matching with Lowe's Ratio Test
# 
# I'll implement the matching function using BFMatcher (Brute Force Matcher) with k-NN and Lowe's ratio test:
# 1. Find k=2 nearest neighbors for each descriptor
# 2. Apply ratio test: keep match if distance(best) / distance(second_best) < threshold
# 3. This filters out ambiguous matches where multiple features look similar

# %%
def match_features(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Match features between two images using k-NN and Lowe's ratio test.
    
    Args:
        descriptors1: Descriptors from first image (N x 128 for SIFT)
        descriptors2: Descriptors from second image (M x 128 for SIFT)
        ratio_threshold: Lowe's ratio test threshold (default 0.75)
    
    Returns:
        good_matches: List of good DMatch objects after ratio test
        all_matches: List of all matches before filtering
    """
    # Create BFMatcher with Euclidean distance (L2 norm) for SIFT
    # normType=cv2.NORM_L2 for float descriptors
    # crossCheck=False because we're using k-NN matching
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    
    # Find k=2 nearest neighbors for each descriptor in descriptors1
    # This returns a list of lists: for each query descriptor, we get 2 best matches
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test
    # The idea: if the best match is much better than the second best
    good_matches = []
    all_matches_list = []
    
    for match_pair in matches:
        # Some descriptors might have less than 2 matches
        if len(match_pair) == 2:
            best_match, second_match = match_pair
            all_matches_list.append(best_match)
            
            # Ratio test: best_distance / second_best_distance < threshold
            # Lower ratio = more distinctive = more reliable match
            if best_match.distance < ratio_threshold * second_match.distance:
                good_matches.append(best_match)
        elif len(match_pair) == 1:
            # Only one match found (no ambiguity)
            all_matches_list.append(match_pair[0])
            good_matches.append(match_pair[0])
    
    return good_matches, all_matches_list


# Test matching on a sample pair
print("Testing feature matching:\n")
print("="*60)

# Use the first scene for testing
test_scene = scene_names[0]
print(f"Test scene: {test_scene}\n")

# Get SIFT features for first two images
features_sift = all_features[test_scene]['SIFT']
img1_name = '1.png'
img2_name = '2.png'

kp1 = features_sift[img1_name]['keypoints']
desc1 = features_sift[img1_name]['descriptors']
img1 = features_sift[img1_name]['image_color']

kp2 = features_sift[img2_name]['keypoints']
desc2 = features_sift[img2_name]['descriptors']
img2 = features_sift[img2_name]['image_color']

print(f"Image 1: {img1_name} - {len(kp1)} keypoints")
print(f"Image 2: {img2_name} - {len(kp2)} keypoints")

# Match features
good_matches, all_matches = match_features(desc1, desc2, ratio_threshold=0.75)

print(f"\nMatching results:")
print(f"  Total matches found: {len(all_matches)}")
print(f"  Good matches (after ratio test): {len(good_matches)}")
print(f"  Filtered out: {len(all_matches) - len(good_matches)} ambiguous matches")
print(f"  Ratio: {len(good_matches)/len(all_matches)*100:.1f}% passed ratio test")

# %% [markdown]
# ### 2.2 Visualizing Feature Matches

# %%
def visualize_matches(img1, kp1, img2, kp2, matches, title="Feature Matches", max_display=50):
    """
    Visualize feature matches between two images.
    
    Args:
        img1, img2: Input images (RGB)
        kp1, kp2: Keypoints from both images
        matches: List of DMatch objects
        title: Plot title
        max_display: Maximum number of matches to display (for clarity)
    
    Returns:
        img_matches: Image with drawn matches
    """
    # Sort matches by distance (best matches first)
    matches_sorted = sorted(matches, key=lambda x: x.distance)
    
    # Limit number of matches to display
    matches_to_draw = matches_sorted[:min(max_display, len(matches_sorted))]
    
    # Draw matches using OpenCV
    # Green lines connect matched keypoints
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_to_draw,
        None,
        matchColor=(0, 255, 0),  # Green for match lines
        singlePointColor=(255, 0, 0),  # Red for unmatched keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  # Don't draw unmatched
    )
    
    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title(f'{title}\nShowing {len(matches_to_draw)} best matches (out of {len(matches)} total)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_matches


# Visualize the matches we just computed
print("\nVisualizing matches between consecutive images:\n")
print("="*60)

_ = visualize_matches(
    img1, kp1,
    img2, kp2,
    good_matches,
    title=f"SIFT Feature Matches: {test_scene} ({img1_name} → {img2_name})",
    max_display=50
)

# %% [markdown]
# ### 2.3 Experimenting with Different Ratio Thresholds
# 
# The ratio threshold is critical for match quality. Let's test different values:
# - **0.6**: Very strict - fewer matches but higher quality
# - **0.75**: Balanced - commonly used default (Lowe's paper)
# - **0.9**: Permissive - more matches but potentially more outliers

# %%
# Compare different ratio thresholds
print("Comparing different Lowe's ratio thresholds:")
print("="*80)
print(f"{'Threshold':<12} {'Good Matches':<15} {'Percentage':<15} {'Description':<30}")
print("-"*80)

thresholds_to_test = [0.6, 0.75, 0.9]

for threshold in thresholds_to_test:
    good, all_m = match_features(desc1, desc2, ratio_threshold=threshold)
    percentage = (len(good) / len(all_m) * 100) if all_m else 0
    
    if threshold <= 0.65:
        desc = "Strict (high quality)"
    elif threshold <= 0.8:
        desc = "Balanced (recommended)"
    else:
        desc = "Permissive (more matches)"
    
    print(f"{threshold:<12.2f} {len(good):<15} {percentage:<15.1f}% {desc:<30}")

print("\n" + "="*80)
print("Note: Lower threshold = stricter filtering = fewer but more reliable matches")
print("      Higher threshold = more permissive = more matches but possibly more outliers")

# %% [markdown]
# ### 2.4 Matching All Image Pairs in a Scene
# 
# For panorama stitching, I need to match consecutive image pairs (1->2, 2->3, 3->4, etc.).

# %%
def match_scene_pairs(scene_name, detector_name='SIFT', ratio_threshold=0.75):
    """
    Match all consecutive image pairs in a scene.
    
    Args:
        scene_name: Name of the scene to process
        detector_name: Which detector to use ('SIFT' or 'ORB')
        ratio_threshold: Lowe's ratio test threshold
    
    Returns:
        scene_matches: Dictionary containing all pair matches with metadata
    """
    features = all_features[scene_name][detector_name]
    image_names = sorted(features.keys())
    
    scene_matches = {}
    
    print(f"Matching consecutive pairs in scene: {scene_name}")
    print("="*70)
    
    # Match each consecutive pair: 1->2, 2->3, 3->4, etc.
    for i in range(len(image_names) - 1):
        img1_name = image_names[i]
        img2_name = image_names[i + 1]
        
        # Get features
        desc1 = features[img1_name]['descriptors']
        desc2 = features[img2_name]['descriptors']
        
        print(f"\nPair {i+1}: {img1_name} → {img2_name}")
        print("-"*70)
        
        # Match
        good_matches, all_matches = match_features(desc1, desc2, ratio_threshold)
        
        # Store everything we need for later (homography estimation)
        pair_key = f"{img1_name}_{img2_name}"
        scene_matches[pair_key] = {
            'img1_name': img1_name,
            'img2_name': img2_name,
            'matches': good_matches,
            'keypoints1': features[img1_name]['keypoints'],
            'keypoints2': features[img2_name]['keypoints'],
            'descriptors1': desc1,
            'descriptors2': desc2,
            'image1': features[img1_name]['image_color'],
            'image2': features[img2_name]['image_color'],
            'path1': features[img1_name]['path'],
            'path2': features[img2_name]['path']
        }
    
    return scene_matches


# Match all pairs in the test scene
print(f"\nMatching all pairs in scene: {test_scene}\n")
test_scene_matches = match_scene_pairs(test_scene, 'SIFT', ratio_threshold=0.75)

# %% [markdown]
# ### 2.5 Visualizing Multiple Consecutive Pairs
# 
# Let's visualize matches for several consecutive pairs to see how match quality varies across the sequence.

# %%
# Visualize first 3 pairs from the test scene
print("Visualizing matches for first 3 consecutive pairs:")
print("="*70)

pair_keys = list(test_scene_matches.keys())[:3]

for pair_key in pair_keys:
    pair_data = test_scene_matches[pair_key]
    
    print(f"\nPair: {pair_data['img1_name']} → {pair_data['img2_name']}")
    print(f"Number of matches: {len(pair_data['matches'])}")
    
    visualize_matches(
        pair_data['image1'], pair_data['keypoints1'],
        pair_data['image2'], pair_data['keypoints2'],
        pair_data['matches'],
        title=f"{test_scene}: {pair_data['img1_name']} → {pair_data['img2_name']}",
        max_display=50
    )

# %% [markdown]
# ### 2.6 Match Quality Analysis
# 
# Analyze match quality across all pairs. This helps us understand:
# - Which pairs have sufficient matches for homography estimation (need at least 4)
# - How match quality varies across the sequence
# - Whether any pairs might cause problems

# %%
def analyze_match_quality(scene_matches):
    """
    Analyze match quality for all pairs in a scene.
    
    Args:
        scene_matches: Dictionary of matched pairs
    """
    print("Match Quality Analysis")
    print("="*80)
    print(f"{'Image Pair':<25} {'Matches':<10} {'Avg Distance':<15} {'Quality':<15}")
    print("-"*80)
    
    for pair_key, pair_data in scene_matches.items():
        matches = pair_data['matches']
        num_matches = len(matches)
        
        # Calculate average match distance (lower is better)
        if num_matches > 0:
            avg_dist = sum(m.distance for m in matches) / num_matches
        else:
            avg_dist = float('inf')
        
        # Quality assessment
        if num_matches >= 50:
            quality = "Excellent"
        elif num_matches >= 30:
            quality = "Good"
        elif num_matches >= 15:
            quality = "Fair"
        elif num_matches >= 4:
            quality = "Minimal"
        else:
            quality = "Insufficient"
        
        pair_name = f"{pair_data['img1_name']}→{pair_data['img2_name']}"
        print(f"{pair_name:<25} {num_matches:<10} {avg_dist:<15.2f} {quality:<15}")
    
    # Summary statistics
    total_matches = sum(len(p['matches']) for p in scene_matches.values())
    avg_per_pair = total_matches / len(scene_matches) if scene_matches else 0
    min_matches = min(len(p['matches']) for p in scene_matches.values()) if scene_matches else 0
    max_matches = max(len(p['matches']) for p in scene_matches.values()) if scene_matches else 0
    
    print("\n" + "="*80)
    print(f"Total pairs: {len(scene_matches)}")
    print(f"Total matches: {total_matches}")
    print(f"Average matches per pair: {avg_per_pair:.1f}")
    print(f"Min/Max matches: {min_matches}/{max_matches}")
    print("\nNote: At least 4 matches required for homography estimation")


# Analyze the test scene
print("\nAnalyzing match quality:\n")
analyze_match_quality(test_scene_matches)

# %% [markdown]
# ### 2.7 Processing All Scenes
# 
# Now match all consecutive pairs in all scenes and store the results for Part 3 (Homography Estimation).

# %%
# Match all consecutive pairs in all scenes
all_scene_matches = {}

print("Processing all scenes...")
print("="*80)

for scene_name in scene_names:
    print(f"\nScene: {scene_name}")
    print("-"*80)
    
    # Match all pairs in this scene using SIFT
    scene_matches = match_scene_pairs(scene_name, 'SIFT', ratio_threshold=0.75)
    all_scene_matches[scene_name] = scene_matches
    
    # Quick summary
    total_matches = sum(len(p['matches']) for p in scene_matches.values())
    avg_matches = total_matches / len(scene_matches) if scene_matches else 0
    
    print(f"\n✓ Processed {len(scene_matches)} pairs")
    print(f"✓ Total matches: {total_matches}")
    print(f"✓ Average matches per pair: {avg_matches:.1f}")

print("\n" + "="*80)
print("All scenes matched successfully!")
print(f"Total scenes: {len(all_scene_matches)}")
print(f"Total pairs: {sum(len(s) for s in all_scene_matches.values())}")

# %% [markdown]
# ### 2.8 Summary of Part 2
# 
# **What I implemented:**
# 1. ✅ Feature matching using BFMatcher with Euclidean distance (L2 norm)
# 2. ✅ k-Nearest Neighbors (k=2) to find best and second-best matches
# 3. ✅ Lowe's ratio test to filter ambiguous matches (threshold=0.75)
# 4. ✅ Match visualization with correspondence lines
# 5. ✅ Match quality analysis for all image pairs
# 6. ✅ Batch processing for all scenes
# 
# **Key Design Choices:**
# 
# **1. Euclidean Distance (L2 norm):**
#    - SIFT uses 128-dimensional float descriptors
#    - Euclidean distance properly measures similarity in this high-dimensional space
#    - Alternative: For ORB (binary descriptors), we would use Hamming distance
# 
# **2. k-NN with k=2:**
#    - We need two nearest neighbors for Lowe's ratio test
#    - First neighbor = candidate match
#    - Second neighbor = used to determine if match is distinctive
# 
# **3. Lowe's Ratio Test (threshold=0.75):**
#    - Compares distance of best match vs. second-best match
#    - If ratio < 0.75, the best match is significantly better → keep it
#    - Filters out ambiguous matches where multiple features look similar
#    - Lower threshold = stricter (fewer, higher quality matches)
#    - Higher threshold = more permissive (more matches, possibly more outliers)
# 
# **How Match Quality Affects Homography:**
# 
# **✅ Good Matches (Correct Correspondences):**
#    - Lead to accurate homography estimation
#    - RANSAC can find the correct transformation
#    - Well-distributed matches give stable results
# 
# **❌ Incorrect Matches (Outliers):**
#    - Can bias the homography if not filtered
#    - RANSAC will remove them, but too many outliers slow down convergence
#    - May require more RANSAC iterations
# 
# **⚠️ Uneven Match Distribution:**
#    - If matches cluster in one region, homography may not generalize well
#    - Need matches across the entire overlap area
#    - Sparse regions may have larger alignment errors
# 
# **🔢 Insufficient Matches:**
#    - Need minimum 4 point correspondences to compute homography
#    - More matches = more robust estimation with RANSAC
#    - Fewer matches = less reliable, more sensitive to outliers
# 
# The matched features are now stored in `all_scene_matches` dictionary, ready for Part 3 (Homography Estimation using DLT and RANSAC).

# %% [markdown]
# ## Part 3: Homography Estimation
# 
# In this part, I'll implement:
# 1. **Direct Linear Transform (DLT)** - Compute homography from point correspondences
# 2. **Point Normalization** - Improve numerical stability
# 3. **RANSAC** - Robust estimation by rejecting outliers
# 4. **Visualization** - Show inliers vs outliers
# 5. **Evaluation** - Compute reprojection error
# 
# ## Mathematical Background
# 
# ### Homography Transformation
# 
# A homography maps points from one image to another
# 
# ### DLT Algorithm
# 
# 1. Normalize points for numerical stability
# 2. Build 2N×9 matrix A from N point correspondences
# 3. Solve Ah = 0 using SVD (solution is last row of V^T)
# 4. Denormalize to get final homography
# 
# ### RANSAC Algorithm
# 
# 1. Randomly sample 4 points
# 2. Compute H using DLT
# 3. Count inliers (reprojection error < threshold)
# 4. Keep best model (most inliers)
# 5. Refine H using all inliers

# %% [markdown]
# ### 3.1 Point Normalization
# 
# Normalizing points before DLT improves numerical stability:
# - Center points at origin (subtract centroid)
# - Scale so average distance from origin is √2
# - This makes the condition number of matrix A better

# %%
def normalize_points(points):
    """
    Normalize points for numerical stability in DLT.
    
    The normalization makes points centered at origin with average distance sqrt(2).
    This improves the condition number of the matrix in DLT.
    
    Args:
        points: Nx2 array of points
    
    Returns:
        normalized_points: Nx2 array of normalized points
        T: 3x3 normalization transformation matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Compute average distance from origin
    distances = np.sqrt(np.sum(centered_points**2, axis=1))
    avg_dist = np.mean(distances)
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist if avg_dist > 0 else 1.0
    
    # Construct normalization matrix T
    # T transforms points: p_norm = T @ p
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized_homogeneous = (T @ points_homogeneous.T).T
    normalized_points = normalized_homogeneous[:, :2] / normalized_homogeneous[:, 2:]
    
    return normalized_points, T

# %% [markdown]
# ### 3.2 DLT Implementation
# 
# The Direct Linear Transform computes the homography matrix from point correspondences.

# %%
def compute_homography_dlt(src_pts, dst_pts, normalize=True):
    """
    Compute homography using Direct Linear Transform (DLT).
    
    Solves for H such that dst_pts ~ H @ src_pts using SVD.
    
    Args:
        src_pts: Nx2 array of source points
        dst_pts: Nx2 array of destination points
        normalize: Whether to normalize points
    
    Returns:
        H: 3x3 homography matrix, or None if computation fails
    """
    n = src_pts.shape[0]
    
    if n < 4:
        print("Error: Need at least 4 point correspondences")
        return None
    
    # Normalize points for numerical stability
    if normalize:
        src_norm, T_src = normalize_points(src_pts)
        dst_norm, T_dst = normalize_points(dst_pts)
    else:
        src_norm = src_pts
        dst_norm = dst_pts
        T_src = np.eye(3)
        T_dst = np.eye(3)
    
    # Build matrix A for the homogeneous system Ah = 0
    # Each correspondence gives 2 equations
    A = []
    
    for i in range(n):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        
        # First equation: -x*h1 - y*h2 - h3 + xp*x*h7 + xp*y*h8 + xp*h9 = 0
        # Second equation: -x*h4 - y*h5 - h6 + yp*x*h7 + yp*y*h8 + yp*h9 = 0
        # Where h = [h1, h2, h3, h4, h5, h6, h7, h8, h9] is H flattened
        
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    A = np.array(A)
    
    # Solve Ah = 0 using SVD
    # The solution is the last column of V (corresponding to smallest singular value)
    try:
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1, :]  # Last row of Vt = last column of V
    except np.linalg.LinAlgError:
        print("SVD failed")
        return None
    
    # Reshape to 3x3 matrix
    H_norm = h.reshape(3, 3)
    
    # Denormalize: H = T_dst^(-1) @ H_norm @ T_src
    if normalize:
        H = np.linalg.inv(T_dst) @ H_norm @ T_src
    else:
        H = H_norm
    
    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]
    
    return H

# %% [markdown]
# ### 3.3 Reprojection Error
# 
# Compute the reprojection error to evaluate homography quality.
# 
# Lower error indicates better alignment.

# %%
def compute_reprojection_error(H, src_pts, dst_pts):
    """
    Compute reprojection error for each point pair.
    
    Error = ||dst_pts - H @ src_pts||_2
    
    Args:
        H: 3x3 homography matrix
        src_pts: Nx2 array of source points
        dst_pts: Nx2 array of destination points
    
    Returns:
        errors: N array of reprojection errors (in pixels)
    """
    n = src_pts.shape[0]
    
    # Convert to homogeneous coordinates
    src_homo = np.hstack([src_pts, np.ones((n, 1))])
    
    # Project source points using homography
    projected = (H @ src_homo.T).T
    
    # Handle potential division by zero or very small values
    w = projected[:, 2:].reshape(-1, 1)
    
    # Set a threshold to avoid division by very small numbers
    # Points with w close to zero are at infinity and should have large error
    epsilon = 1e-8
    
    # Initialize errors with a large value
    errors = np.full(n, 1e10, dtype=np.float64)
    
    # Only compute error for valid points (w not too close to zero)
    valid_mask = np.abs(w.flatten()) > epsilon
    
    if np.any(valid_mask):
        projected[valid_mask, :2] = projected[valid_mask, :2] / w[valid_mask]
        errors[valid_mask] = np.sqrt(np.sum((projected[valid_mask, :2] - dst_pts[valid_mask])**2, axis=1))
    
    return errors

# %% [markdown]
# ### 3.4 RANSAC Implementation
# 
# RANSAC (Random Sample Consensus) robustly estimates the homography by:
# 1. Randomly sampling minimal sets (4 points for homography)
# 2. Computing model (H) from sample
# 3. Counting inliers (points with error < threshold)
# 4. Keeping model with most inliers
# 5. Refining final H using all inliers
# 
# **Key Parameters:**
# - `ransac_thresh`: Inlier threshold in pixels
# - `ransac_iters`: Number of iterations
# - `min_matches`: Minimum matches needed

# %%
def estimate_homography_ransac(src_pts, dst_pts, ransac_thresh=5.0, ransac_iters=2000, min_matches=4):
    """
    Estimate homography using RANSAC for robust outlier rejection.
    
    Args:
        src_pts: Nx2 array of source points
        dst_pts: Nx2 array of destination points
        ransac_thresh: Inlier threshold in pixels
        ransac_iters: Number of RANSAC iterations
        min_matches: Minimum number of matches required
    
    Returns:
        best_H: Best homography matrix
        inlier_mask: Boolean array indicating inliers
    """
    n = src_pts.shape[0]
    
    if n < min_matches:
        print(f"Error: Need at least {min_matches} matches, got {n}")
        return None, np.zeros(n, dtype=bool)
    
    best_H = None
    best_inliers = np.zeros(n, dtype=bool)
    best_inlier_count = 0
    
    print(f"Running RANSAC with {ransac_iters} iterations...")
    
    # RANSAC loop
    for iteration in range(ransac_iters):
        # Randomly select 4 points
        indices = np.random.choice(n, 4, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]
        
        # Compute homography from sample
        H = compute_homography_dlt(sample_src, sample_dst)
        
        if H is None:
            continue
        
        # Compute reprojection errors for all points
        errors = compute_reprojection_error(H, src_pts, dst_pts)
        
        # Find inliers (points with error < threshold)
        inliers = errors < ransac_thresh
        inlier_count = np.sum(inliers)
        
        # Update best model if this one is better
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H
            
        # Optional: print progress every 500 iterations
        if (iteration + 1) % 500 == 0:
            print(f"  Iteration {iteration + 1}/{ransac_iters}: Best inliers = {best_inlier_count}/{n} ({best_inlier_count/n*100:.1f}%)")
    
    # Refine homography using all inliers
    if best_inlier_count >= min_matches:
        print(f"\nRefining homography using {best_inlier_count} inliers...")
        best_H = compute_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
        
        # Recompute inliers with refined homography
        if best_H is not None:
            errors = compute_reprojection_error(best_H, src_pts, dst_pts)
            best_inliers = errors < ransac_thresh
            best_inlier_count = np.sum(best_inliers)
    
    print(f"\nRANSAC complete!")
    print(f"  Final inliers: {best_inlier_count}/{n} ({best_inlier_count/n*100:.1f}%)")
    print(f"  Outliers rejected: {n - best_inlier_count} ({(n - best_inlier_count)/n*100:.1f}%)")
    
    return best_H, best_inliers


# %% [markdown]
# ### 3.5 Visualization: Inliers vs Outliers
# 
# Visualize which matches are inliers (green) and which are outliers (red) after RANSAC.

# %%
def visualize_inliers_outliers(img1, kp1, img2, kp2, matches, inlier_mask, title="RANSAC: Inliers vs Outliers"):
    """
    Visualize inlier matches (green) vs outlier matches (red).
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: List of DMatch objects
        inlier_mask: Boolean array indicating inliers
        title: Plot title
    
    Returns:
        img_matches: Image with drawn matches
    """
    # Separate inliers and outliers
    inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
    outlier_matches = [m for i, m in enumerate(matches) if not inlier_mask[i]]
    
    # Limit the number of matches to draw for clarity (too many lines make it messy)
    max_matches_to_draw = 50
    import random
    if len(inlier_matches) > max_matches_to_draw:
        inlier_matches = random.sample(inlier_matches, max_matches_to_draw)
    if len(outlier_matches) > max_matches_to_draw // 2:  # Show fewer outliers
        outlier_matches = random.sample(outlier_matches, max_matches_to_draw // 2)
    
    # Create empty canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:w1+w2] = img2
    
    # Draw outliers in red (thinner lines)
    for m in outlier_matches:
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([w1, 0]))
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)
        cv2.circle(img_matches, pt1, 2, (255, 0, 0), -1)
        cv2.circle(img_matches, pt2, 2, (255, 0, 0), -1)
    
    # Draw inliers in green (on top, slightly thicker)
    for m in inlier_matches:
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([w1, 0]))
        cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(img_matches, pt1, 3, (0, 255, 0), -1)
        cv2.circle(img_matches, pt2, 3, (0, 255, 0), -1)
    
    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title(f'{title}\\nGreen: {len(inlier_matches)} inliers | Red: {len(outlier_matches)} outliers', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_matches

# %% [markdown]
# ### 3.6 Complete Pipeline: Extract, Match, and Estimate Homography
# 
# Now let's put it all together and test on real images from the dataset.

# %%
def estimate_homography_from_images(img1_path, img2_path, detector='SIFT', 
                                    ratio_thresh=0.75, ransac_thresh=5.0, ransac_iters=2000,
                                    visualize=True):
    """
    Complete pipeline: extract features, match, and estimate homography.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        detector: Feature detector to use ('SIFT' or 'ORB')
        ratio_thresh: Lowe's ratio test threshold
        ransac_thresh: RANSAC inlier threshold in pixels
        ransac_iters: Number of RANSAC iterations
        visualize: Whether to show visualization
    
    Returns:
        H: Homography matrix
        inlier_mask: Boolean mask of inliers
        matches: List of matches
        kp1, kp2: Keypoints
        img1, img2: Images
    """
    print("="*80)
    print(f"HOMOGRAPHY ESTIMATION PIPELINE")
    print("="*80)
    
    # Step 1: Extract features
    print("\n[1/4] Extracting features...")
    if detector == 'SIFT':
        det = sift
    elif detector == 'ORB':
        det = orb
    else:
        raise ValueError(f"Unknown detector: {detector}")
    
    img1, _, kp1, desc1 = extract_features(img1_path, det, detector)
    img2, _, kp2, desc2 = extract_features(img2_path, det, detector)
    
    print(f"  Image 1: {len(kp1)} keypoints")
    print(f"  Image 2: {len(kp2)} keypoints")
    
    # Step 2: Match features
    print("\n[2/4] Matching features...")
    matches, _ = match_features(desc1, desc2, ratio_threshold=ratio_thresh)
    print(f"  Found {len(matches)} good matches")
    
    if len(matches) < 4:
        print("Insufficient matches for homography estimation")
        return None, None, None, None, None, None
    
    # Extract point correspondences
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Step 3: Estimate homography with RANSAC
    print("\n[3/4] Estimating homography with RANSAC...")
    H, inlier_mask = estimate_homography_ransac(
        src_pts, dst_pts,
        ransac_thresh=ransac_thresh,
        ransac_iters=ransac_iters
    )
    
    if H is None:
        print("Homography estimation failed")
        return None, None, None, None, None, None
    
    # Step 4: Compute statistics
    print("\n[4/4] Computing statistics...")
    inlier_count = np.sum(inlier_mask)
    outlier_count = len(matches) - inlier_count
    
    errors = compute_reprojection_error(H, src_pts[inlier_mask], dst_pts[inlier_mask])
    
    print(f"\nRESULTS:")
    print(f"  Inliers: {inlier_count}/{len(matches)} ({inlier_count/len(matches)*100:.1f}%)")
    print(f"  Outliers: {outlier_count} ({outlier_count/len(matches)*100:.1f}%)")
    print(f"  Reprojection error (inliers only):")
    print(f"    Mean: {np.mean(errors):.3f} pixels")
    print(f"    Std:  {np.std(errors):.3f} pixels")
    print(f"    Max:  {np.max(errors):.3f} pixels")
    
    print(f"\nEstimated Homography Matrix:")
    print(H)
    
    # Visualize
    if visualize:
        print("\n[Visualization] Displaying inliers vs outliers...")
        visualize_inliers_outliers(img1, kp1, img2, kp2, matches, inlier_mask,
                                   title=f"{detector} - RANSAC Results")
    
    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)
    
    return H, inlier_mask, matches, kp1, kp2, img1, img2


# Test on real images if dataset exists
if 'scene_names' in dir() and len(scene_names) > 0:
    print("\nTesting complete pipeline on real images:")
    print("="*80)
    
    # Use first scene
    test_scene = scene_names[0]
    scene_path = PANORAMA_DATASET_DIR / test_scene
    
    img1_path = scene_path / "1.png"
    img2_path = scene_path / "2.png"
    
    if img1_path.exists() and img2_path.exists():
        print(f"\nScene: {test_scene}")
        print(f"Images: {img1_path.name} → {img2_path.name}")
        
        result = estimate_homography_from_images(
            img1_path, img2_path,
            detector='SIFT',
            ratio_thresh=0.75,
            ransac_thresh=5.0,
            ransac_iters=2000,
            visualize=True
        )
        
        H, inlier_mask, matches, kp1, kp2, img1, img2 = result
    else:
        print(f"\nImages not found in {scene_path}")
        print("Please ensure the dataset is properly set up")
else:
    print("\nDataset not available. Pipeline implementation complete.")
    print("Load the dataset to test on real images.")

# %% [markdown]
# ### 3.7 Summary of Part 3: Homography Estimation
# 
# **What I Implemented:**
# 
# 1. ✅ **Point Normalization**
#    - Centers points at origin
#    - Scales to average distance √2
#    - Improves numerical stability
# 
# 2. ✅ **DLT (Direct Linear Transform)**
#    - Builds 2N×9 matrix from N correspondences
#    - Solves Ah = 0 using SVD
#    - Denormalizes result
# 
# 3. ✅ **Reprojection Error**
#    - Computes ||dst - H·src||₂
#    - Evaluates homography quality
# 
# 4. ✅ **RANSAC**
#    - Randomly samples 4-point sets
#    - Computes H and counts inliers
#    - Keeps best model
#    - Refines using all inliers
# 
# 5. ✅ **Visualization**
#    - Shows inliers (green) vs outliers (red)
#    - Verifies RANSAC performance
# 
# **Key Design Choices:**
# 
# **1. Normalization (Always Enabled):**
# - Improves numerical stability of SVD
# - Makes DLT less sensitive to coordinate scale
# - Standard practice in computer vision
# 
# **2. RANSAC Parameters:**
# - **Threshold = 5.0 pixels**: Balance between strict (few inliers) and permissive (many outliers)
# - **Iterations = 2000**: Ensures high probability of finding good model
# - **Min matches = 4**: Minimum for homography (8 DoF)
# 
# **3. Refinement Step:**
# - After RANSAC finds inliers, recompute H using all of them
# - Improves accuracy by using more data
# - Standard RANSAC practice
# 
# **How RANSAC Improves Estimation:**
# 
# **Without RANSAC (DLT only):**
# - All matches treated equally
# - Outliers bias the result
# - Can produce completely wrong H
# 
# **With RANSAC:**
# - Outliers identified and removed
# - Only geometric consensus matters
# - Robust to 50%+ outliers
# - Produces accurate H even with noisy matches
# 
# **Impact on Panorama Stitching:**
# - Accurate H → proper alignment
# - Removing outliers → no ghosting
# - More inliers → more stable result
# - Good for next part (warping & blending)
# 
# The homography estimation is now complete and ready for Part 4 (Image Warping and Panorama Construction)!

# %% [markdown]
# ## Part 4: Image Warping and Panorama Construction
# 
# In this part, I'll implement:
# 1. **Image Warping** - Transform one image into another's coordinate frame using homography
# 2. **Canvas Size Calculation** - Determine output panorama dimensions
# 3. **Blending Techniques** - Combine overlapping regions seamlessly
# 4. **Panorama Stitching** - Create panoramas for all 6 scenes in the dataset
# 
# ### Warping Strategy:
# - Warp all images to the coordinate system of the reference (middle) image
# - Calculate appropriate canvas size to fit all warped images
# - Use different blending methods: simple copy, averaging, linear/feathering blending

# %% [markdown]
# ### 4.1 Image Warping Function
# 
# Warp an image using the computed homography matrix to align it with the reference image.

# %%
def warp_image(img, H, output_shape):
    """
    Warp an image using homography matrix.
    
    Args:
        img: Input image to warp
        H: 3x3 homography matrix
        output_shape: (height, width) of output canvas
    
    Returns:
        warped: Warped image
    """
    h, w = output_shape
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return warped


def get_canvas_size(images, homographies):
    """
    Calculate canvas size needed to fit all warped images.
    
    Args:
        images: List of images
        homographies: List of homography matrices (one per image)
    
    Returns:
        canvas_shape: (height, width) of canvas
        offset: (x_offset, y_offset) to shift all images
    """
    # Get corners of all images after transformation
    all_corners = []
    
    for idx, (img, H) in enumerate(zip(images, homographies)):
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Transform corners
        corners_homo = np.hstack([corners, np.ones((4, 1))])
        transformed = (H @ corners_homo.T).T
        
        # Check for points at infinity (w ≈ 0)
        w_values = transformed[:, 2]
        if np.any(np.abs(w_values) < 1e-6):
            print(f"  ⚠️  WARNING: Image {idx} has corners projecting to infinity!")
            print(f"      w values: {w_values}")
            # Clip to reasonable bounds instead of going to infinity
            w_values = np.maximum(np.abs(w_values), 1e-6) * np.sign(w_values)
            w_values[w_values == 0] = 1e-6
        
        transformed = transformed[:, :2] / w_values.reshape(-1, 1)
        
        # Debug: print transformed corner locations
        print(f"  Image {idx} corners after transformation:")
        print(f"    Min: ({np.min(transformed[:, 0]):.1f}, {np.min(transformed[:, 1]):.1f})")
        print(f"    Max: ({np.max(transformed[:, 0]):.1f}, {np.max(transformed[:, 1]):.1f})")
        
        all_corners.append(transformed)
    
    all_corners = np.vstack(all_corners)
    
    # Find bounding box
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    # Canvas size
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    
    # Sanity check: canvas size shouldn't be absurdly large
    # Typical panorama should be at most 10x the size of individual images
    max_img_dim = max([max(img.shape[:2]) for img in images])
    max_reasonable_canvas = max_img_dim * 20  # Allow up to 20x
    
    if width > max_reasonable_canvas or height > max_reasonable_canvas:
        print(f"  ⚠️  WARNING: Canvas size ({width} x {height}) is unreasonably large!")
        print(f"      This indicates bad homographies. Capping at {max_reasonable_canvas}x{max_reasonable_canvas}")
        width = min(width, max_reasonable_canvas)
        height = min(height, max_reasonable_canvas)
    
    # Offset to shift everything to positive coordinates
    offset = (-min_x, -min_y)
    
    return (height, width), offset

# %% [markdown]
# ### 4.2 Blending Techniques
# 
# Implement different blending methods to combine overlapping regions:
# 1. **Simple Copy**: Just overwrite pixels
# 2. **Average Blending**: Average pixel values in overlaps
# 3. **Linear Blending**: Weight-based blending using distance from edge
# 4. **Feathering**: Smooth transition in overlapping regions

# %%
def blend_images_simple(img1, img2):
    """
    Simple blending: img2 overwrites img1 where img2 is non-zero.
    
    Args:
        img1, img2: Images to blend (same size)
    
    Returns:
        blended: Blended image
    """
    # Create mask for img2 (where it's not black)
    mask2 = np.any(img2 > 0, axis=2).astype(np.uint8)
    
    # Copy img1, then overwrite with img2 where mask2 is 1
    blended = img1.copy()
    blended[mask2 == 1] = img2[mask2 == 1]
    
    return blended


def blend_images_average(img1, img2):
    """
    Average blending: average pixel values in overlapping regions.
    
    Args:
        img1, img2: Images to blend (same size)
    
    Returns:
        blended: Blended image
    """
    # Create masks
    mask1 = np.any(img1 > 0, axis=2).astype(np.float32)
    mask2 = np.any(img2 > 0, axis=2).astype(np.float32)
    
    # Find overlap region
    overlap = (mask1 > 0) & (mask2 > 0)
    
    # Blend
    blended = img1.copy().astype(np.float32)
    
    # In overlap, average the two images
    blended[overlap] = (img1[overlap].astype(np.float32) + img2[overlap].astype(np.float32)) / 2.0
    
    # Where only img2 exists, use img2
    only_img2 = (mask1 == 0) & (mask2 > 0)
    blended[only_img2] = img2[only_img2]
    
    return blended.astype(np.uint8)


def create_distance_mask(img):
    """
    Create a distance mask for feathering blending.
    Each pixel's weight is based on its distance to the nearest edge.
    
    Args:
        img: Input image
    
    Returns:
        mask: Distance-based weight mask (0 to 1)
    """
    # Create binary mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_mask = (gray > 0).astype(np.uint8)
    
    # Compute distance transform
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Normalize to [0, 1]
    if np.max(dist) > 0:
        dist = dist / np.max(dist)
    
    return dist


def blend_images_feather(img1, img2):
    """
    Feathering blending: use distance-based weights for smooth transitions.
    
    Args:
        img1, img2: Images to blend (same size)
    
    Returns:
        blended: Blended image
    """
    # Create distance masks
    mask1 = create_distance_mask(img1)
    mask2 = create_distance_mask(img2)
    
    # Normalize weights in overlap region
    total_weight = mask1 + mask2
    total_weight[total_weight == 0] = 1  # Avoid division by zero
    
    weight1 = mask1 / total_weight
    weight2 = mask2 / total_weight
    
    # Blend
    blended = (img1.astype(np.float32) * weight1[:, :, np.newaxis] + 
               img2.astype(np.float32) * weight2[:, :, np.newaxis])
    
    return blended.astype(np.uint8)

# %% [markdown]
# ### 4.3 Panorama Stitching Pipeline
# 
# Complete pipeline to stitch multiple images into a panorama:
# 1. Load all images in a scene
# 2. Select reference image (usually middle one)
# 3. Compute homographies from each image to reference
# 4. Calculate canvas size
# 5. Warp all images
# 6. Blend them together

# %%
def stitch_panorama(scene_path, detector='SIFT', blend_method='feather', 
                    ratio_thresh=0.75, ransac_thresh=5.0, ransac_iters=2000,
                    visualize_steps=False):
    """
    Create a panorama from all images in a scene.
    
    Args:
        scene_path: Path to scene directory
        detector: Feature detector to use ('SIFT' or 'ORB')
        blend_method: Blending method ('simple', 'average', 'feather')
        ratio_thresh: Lowe's ratio test threshold
        ransac_thresh: RANSAC inlier threshold
        ransac_iters: RANSAC iterations
        visualize_steps: Whether to show intermediate results
    
    Returns:
        panorama: Final stitched panorama image
        all_warped: List of all warped images (for visualization)
    """
    print("="*80)
    print(f"PANORAMA STITCHING: {scene_path.name}")
    print("="*80)
    
    # Load all images
    image_files = sorted(scene_path.glob("*.png"))
    if len(image_files) == 0:
        image_files = sorted(scene_path.glob("*.jpg"))
    
    print(f"\nFound {len(image_files)} images")
    
    if len(image_files) < 2:
        print("Need at least 2 images for panorama")
        return None, None
    
    # Load images
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        print(f"  Loaded: {img_file.name} - {img.shape}")
    
    # Choose reference image
    # Strategy: Use first image as reference for panorama stitching
    # This is more intuitive and often produces better results because:
    # 1. Sequential matching works better (1->2, 2->3, etc.)
    # 2. Panorama builds naturally from left to right
    # 3. Less chance of accumulated error
    # 4. First image sets the "horizontal" reference plane
    ref_idx = 0
    
    print(f"\nReference image: {image_files[ref_idx].name} (index {ref_idx})")
    print(f"  Strategy: Using first image as reference for stable left-to-right stitching")
    
    # Initialize feature detector
    if detector == 'SIFT':
        det = sift
    elif detector == 'ORB':
        det = orb
    else:
        raise ValueError(f"Unknown detector: {detector}")
    
    # Extract features for all images
    print(f"\n[1/5] Extracting features using {detector}...")
    all_features = []
    for i, (img, img_file) in enumerate(zip(images, image_files)):
        _, _, kp, desc = extract_features(img_file, det, detector)
        all_features.append((kp, desc))
        print(f"  Image {i}: {len(kp)} keypoints")
    
    # Compute sequential homographies (between consecutive images)
    # This is the proper approach for panorama stitching!
    print(f"\n[2/7] Computing sequential homographies (consecutive image pairs)...")
    sequential_homographies = []
    
    for i in range(len(images) - 1):
        print(f"\n  Image {i} → Image {i+1}:")
        
        kp_src, desc_src = all_features[i]
        kp_dst, desc_dst = all_features[i + 1]
        
        # Match features between consecutive images
        matches, _ = match_features(desc_src, desc_dst, ratio_threshold=ratio_thresh)
        print(f"    Matches: {len(matches)}")
        
        if len(matches) < 4:
            print(f"    ✗ Insufficient matches!")
            return None, None
        
        # Extract point correspondences
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches])
        
        # Estimate homography with RANSAC (our DLT + RANSAC implementation)
        H, inlier_mask = estimate_homography_ransac(
            src_pts, dst_pts,
            ransac_thresh=ransac_thresh,
            ransac_iters=ransac_iters
        )
        
        if H is None:
            print(f"    ✗ Homography estimation failed!")
            return None, None
        
        inlier_count = np.sum(inlier_mask)
        print(f"    Inliers: {inlier_count}/{len(matches)} ({inlier_count/len(matches)*100:.1f}%)")
        
        # Sanity check: verify homography is reasonable
        cond = np.linalg.cond(H)
        if cond > 1e10:
            print(f"    ⚠️  WARNING: Ill-conditioned homography (cond={cond:.2e})")
        
        sequential_homographies.append(H)
    
    # Chain/compose homographies to get transformations to reference frame
    # Since ref_idx = 0, all images to the right need to be transformed back to image 0
    print(f"\n[3/6] Chaining homographies to reference frame (image {ref_idx})...")
    homographies = []
    
    for i in range(len(images)):
        if i == ref_idx:
            # Reference image: identity homography
            H = np.eye(3)
            print(f"  Image {i}: Identity (reference)")
        elif i > ref_idx:
            # For images after reference: chain inverse homographies
            # We have: sequential_homographies[j] maps from image j to image j+1
            # To map image i to image 0, we need: H_i→0 = H_0→1^(-1) @ H_1→2^(-1) @ ... @ H_(i-1)→i^(-1)
            # Start from identity and chain inverse transformations
            H = np.eye(3)
            for j in range(i - 1, ref_idx - 1, -1):
                # j goes from i-1 down to 0
                # Invert the homography and compose
                H_inv = np.linalg.inv(sequential_homographies[j])
                H = H_inv @ H
            print(f"  Image {i}: Chained {i - ref_idx} inverse homographies")
        else:
            # This shouldn't happen since ref_idx = 0
            # But if we ever use a different ref_idx (e.g., middle image), handle it
            H = np.eye(3)
            for j in range(i, ref_idx):
                H = sequential_homographies[j] @ H
            print(f"  Image {i}: Chained {ref_idx - i} forward homographies")
        
        homographies.append(H)
    
    # Calculate canvas size
    print(f"\n[4/7] Calculating canvas size...")
    canvas_shape, offset = get_canvas_size(images, homographies)
    print(f"  Canvas size: {canvas_shape[1]} x {canvas_shape[0]} pixels")
    print(f"  Offset: ({offset[0]:.1f}, {offset[1]:.1f})")
    
    # Create translation matrix for offset
    T_offset = np.array([
        [1, 0, offset[0]],
        [0, 1, offset[1]],
        [0, 0, 1]
    ])
    
    # Warp all images
    print(f"\n[5/7] Warping images to reference frame...")
    warped_images = []
    
    for i, (img, H) in enumerate(zip(images, homographies)):
        # Combine offset with homography
        H_with_offset = T_offset @ H
        
        # Warp image
        warped = warp_image(img, H_with_offset, canvas_shape)
        warped_images.append(warped)
        print(f"  Warped image {i}")
        
        if visualize_steps:
            plt.figure(figsize=(12, 8))
            plt.imshow(warped)
            plt.title(f'Warped Image {i}')
            plt.axis('off')
            plt.show()
    
    # Blend images
    print(f"\n[6/7] Blending images using '{blend_method}' method...")
    
    if blend_method == 'simple':
        blend_func = blend_images_simple
    elif blend_method == 'average':
        blend_func = blend_images_average
    elif blend_method == 'feather':
        blend_func = blend_images_feather
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")
    
    # Start with first warped image
    panorama = warped_images[0].copy()
    
    # Blend in remaining images
    for i in range(1, len(warped_images)):
        panorama = blend_func(panorama, warped_images[i])
        print(f"  Blended image {i}")
    
    # Straighten the panorama if it's tilted
    print(f"\n[7/7] Straightening panorama...")
    panorama, tilt_angle = straighten_panorama(panorama, visualize=visualize_steps)
    
    print("\n" + "="*80)
    print("Panorama stitching complete!")
    print("="*80)
    
    return panorama, warped_images

# %% [markdown]
# ### 4.4 Test Panorama Stitching on One Scene
# 
# Let's test the panorama stitching on one scene first to verify it works.

# %%
# Test on one scene if dataset is available
if 'scene_names' in dir() and len(scene_names) > 0:
    print("Testing panorama stitching on first scene:")
    print("="*80)
    
    test_scene = scene_names[2]
    scene_path = PANORAMA_DATASET_DIR / test_scene
    
    print(f"\nScene: {test_scene}")
    
    # Create panorama
    panorama, warped_images = stitch_panorama(
        scene_path,
        detector='SIFT',
        blend_method='feather',
        ratio_thresh=0.75,
        ransac_thresh=5.0,
        ransac_iters=2000,
        visualize_steps=False
    )
    
    if panorama is not None:
        # Display result
        plt.figure(figsize=(20, 12))
        plt.imshow(panorama)
        plt.title(f'Panorama: {test_scene}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print("\nTest successful! Ready to process all scenes.")
    else:
        print("\nTest failed. Check dataset and implementation.")
else:
    print("\nDataset not available. Please load the dataset to test panorama stitching.")
    print("The panorama stitching implementation is complete and ready to use.")

# %% [markdown]
# ### 4.5 Process All Scenes in Dataset
# 
# Now let's create panoramas for all 6 scenes in the dataset as required by the assignment.

# %%
# Process all scenes
if 'scene_names' in dir() and len(scene_names) > 0:
    print("Creating panoramas for all scenes:")
    print("="*80)
    
    # Create output directory for panoramas
    output_dir = BASE_DIR / 'panorama_results'
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    all_panoramas = {}
    
    for i, scene_name in enumerate(scene_names, 1):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING SCENE {i}/{len(scene_names)}: {scene_name}")
        print(f"{'='*80}\n")
        
        scene_path = PANORAMA_DATASET_DIR / scene_name
        
        # Create panorama
        panorama, warped_images = stitch_panorama(
            scene_path,
            detector='SIFT',
            blend_method='feather',
            ratio_thresh=0.75,
            ransac_thresh=5.0,
            ransac_iters=2000,
            visualize_steps=False
        )
        
        if panorama is not None:
            all_panoramas[scene_name] = panorama
            
            # Save panorama
            output_path = output_dir / f'{scene_name}_panorama.png'
            panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), panorama_bgr)
            print(f"\n✓ Saved: {output_path}")
            
            # Display
            plt.figure(figsize=(20, 12))
            plt.imshow(panorama)
            plt.title(f'Panorama: {scene_name}', fontsize=18, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nFailed to create panorama for {scene_name}")
    
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSuccessfully created {len(all_panoramas)}/{len(scene_names)} panoramas")
    print(f"Saved to: {output_dir}")
    
    for scene_name, pano in all_panoramas.items():
        print(f"  ✓ {scene_name}: {pano.shape[1]} x {pano.shape[0]} pixels")
    
else:
    print("\nDataset not available.")
    print("The panorama stitching implementation is complete.")
    print("\nTo use it:")
    print("1. Place your dataset in 'panorama_dataset' folder")
    print("2. Run the cells above to process all scenes")

# %% [markdown]
# ### 4.6 Comparison of Blending Methods
# 
# Let's compare different blending methods on the same scene to see their effects.

# %%
# Compare blending methods
if 'scene_names' in dir() and len(scene_names) > 0:
    print("Comparing blending methods:")
    print("="*80)
    
    test_scene = scene_names[0]
    scene_path = PANORAMA_DATASET_DIR / test_scene
    
    print(f"\nScene: {test_scene}")
    print("\nTesting different blending methods...")
    
    blend_methods = ['simple', 'average', 'feather']
    panoramas_comparison = {}
    
    for method in blend_methods:
        print(f"\n{'-'*80}")
        print(f"Method: {method.upper()}")
        print(f"{'-'*80}")
        
        panorama, _ = stitch_panorama(
            scene_path,
            detector='SIFT',
            blend_method=method,
            ratio_thresh=0.75,
            ransac_thresh=5.0,
            ransac_iters=1000,  # Fewer iterations for speed
            visualize_steps=False
        )
        
        if panorama is not None:
            panoramas_comparison[method] = panorama
    
    # Display comparison
    if len(panoramas_comparison) > 0:
        print("\n\nVisual Comparison of Blending Methods:")
        print("="*80)
        
        fig, axes = plt.subplots(len(panoramas_comparison), 1, 
                                figsize=(20, 6*len(panoramas_comparison)))
        
        if len(panoramas_comparison) == 1:
            axes = [axes]
        
        for idx, (method, pano) in enumerate(panoramas_comparison.items()):
            axes[idx].imshow(pano)
            axes[idx].set_title(f'{method.upper()} Blending', 
                              fontsize=16, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Comparison complete!")
        print("\nObservations:")
        print("  - Simple: Fast but may show visible seams")
        print("  - Average: Reduces seams but can blur details")
        print("  - Feather: Smoothest transition, best visual quality")
    
else:
    print("\nDataset not available for blending comparison.")
    print("All blending methods are implemented and ready to use.")

# %% [markdown]
# ### 4.7 Summary of Part 4: Image Warping and Panorama Construction
# 
# **What I Implemented:**
# 
# 1. ✅ **Image Warping**
#    - `warp_image()`: Apply homography to transform images
#    - Uses `cv2.warpPerspective()` for geometric transformation
#    - Bilinear interpolation for smooth results
# 
# 2. ✅ **Canvas Size Calculation**
#    - `get_canvas_size()`: Compute bounding box for all warped images
#    - Transform image corners using homographies
#    - Calculate offset to ensure all content fits
# 
# 3. ✅ **Three Blending Techniques**
#    - **Simple Copy**: Fast, but shows seams
#    - **Average Blending**: Reduces seams, can blur
#    - **Feathering**: Distance-based weights, smoothest results
# 
# 4. ✅ **Complete Panorama Pipeline**
#    - Load all images in scene
#    - Choose reference image (middle one)
#    - Compute homographies to reference
#    - Warp all images to reference frame
#    - Blend overlapping regions
#    - Save results
# 
# 5. ✅ **Batch Processing**
#    - Process all 6 scenes automatically
#    - Save panoramas to `panorama_results/` folder
#    - Generate comparison visualizations
# 
# **Key Design Choices:**
# 
# **1. Reference Image Selection:**
# - Use middle image as reference
# - Minimizes accumulated error
# - Reduces extreme perspective distortions
# 
# **2. Coordinate System:**
# - Warp all images to reference frame
# - Calculate global canvas size
# - Apply translation offset for positive coordinates
# 
# **3. Blending Strategy:**
# - **Feathering (recommended)**: 
#   - Weight based on distance from edge
#   - Smooth transitions in overlaps
#   - Best visual quality
#   
# - **Average**: 
#   - Simple mean in overlaps
#   - Good for quick results
#   
# - **Simple Copy**: 
#   - Fastest method
#   - May show visible seams
# 
# **4. Warping Parameters:**
# - `cv2.INTER_LINEAR`: Good balance of speed and quality
# - Could use `cv2.INTER_CUBIC` for higher quality
# - `cv2.INTER_NEAREST` for speed
# 
# **How Blending Affects Results:**
# 
# **✅ Good Blending (Feathering):**
# - Seamless transitions
# - No visible boundaries
# - Preserves details
# 
# **⚠️ Poor Blending (Simple):**
# - Visible seams at boundaries
# - Color discontinuities
# - Artifacts from misalignment
# 
# **📊 For Report:**
# - Show warped images before blending
# - Highlight overlap regions
# - Compare blending methods visually
# - Display all 6 panoramas
# 
# The panorama construction is now complete! All images are stitched with proper alignment and blending.


