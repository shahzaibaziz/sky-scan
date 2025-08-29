#!/usr/bin/env python3
"""
Quick setup test for roof texture classification
Tests paths, finds available files, and provides next steps
"""

import os
import sys

def test_paths():
    """Test if the correct paths exist"""
    print("🔍 Testing Paths...")

    base_path = "/home/student/sky-scan/notebooks"
    data_path = "/home/student/sky-scan/data"

    paths_to_check = {
        "Base notebooks directory": base_path,
        "Data directory": data_path,
        "Training data patches": f"{data_path}/patch",
        "Binary patches": f"{data_path}/patch-binary",
        "Color patches": f"{data_path}/patch-texture"
    }

    print("\n📂 Path Status:")
    all_good = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_good = False

    return all_good

def find_images():
    """Find available images for testing"""
    print("\n🖼️  Finding Images...")

    image_paths = [
        "/home/student/sky-scan/notebooks/output.png",
        "/home/student/sky-scan/notebooks/learning graph 17 augest.png",
        "/home/student/sky-scan/notebooks/with_3_classes_imrpove_cloudav1.png"
    ]

    found_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            found_images.append(img_path)
            print(f"  ✅ Found: {img_path}")
        else:
            print(f"  ❌ Missing: {img_path}")

    # Also check current directory
    try:
        current_files = os.listdir(".")
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        local_images = [f for f in current_files if any(f.lower().endswith(ext) for ext in image_extensions)]

        if local_images:
            print(f"\n  📁 Local images found:")
            for img in local_images:
                print(f"    • {img}")
                found_images.append(os.path.abspath(img))
    except:
        pass

    return found_images

def find_models():
    """Find available trained models"""
    print("\n🤖 Finding Models...")

    model_paths = [
        "/home/student/sky-scan/notebooks/models/roof_texture_model.pth",
        "/home/student/sky-scan/notebooks/models/conservative_roof_texture_model.pth",
        "/home/student/sky-scan/notebooks/models/improved_roof_texture_model.pth",
        "/home/student/sky-scan/notebooks/best_roof_classifier.pth",
        "models/roof_texture_model.pth",
        "best_roof_classifier.pth"
    ]

    found_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            found_models.append(model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✅ Found: {model_path} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ Missing: {model_path}")

    return found_models

def check_notebooks():
    """Check available notebook files"""
    print("\n📓 Available Notebooks:")

    notebook_files = [
        "with_3_classes_improve_from_personal_cloudaV2.ipynb",  # Training
        "model_inference_visualization.ipynb",  # Inference
        "roof_inference_clean.ipynb",  # Alternative inference
    ]

    for notebook in notebook_files:
        if os.path.exists(notebook):
            print(f"  ✅ {notebook}")
        else:
            print(f"  ❌ {notebook}")

def provide_next_steps(found_models, found_images, paths_ok):
    """Provide clear next steps based on what's available"""
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)

    if not paths_ok:
        print("❌ PATH ISSUE:")
        print("  Some required directories are missing.")
        print("  Make sure you're in the correct environment:")
        print("  cd /home/student/sky-scan/notebooks")
        return

    if found_models:
        print("✅ MODELS AVAILABLE - Ready for Inference!")
        print(f"  Found {len(found_models)} trained model(s)")
        if found_images:
            print(f"  Found {len(found_images)} test image(s)")
            print("\n🚀 RUN INFERENCE:")
            print("  1. Open: model_inference_visualization.ipynb")
            print("  2. Run all cells")
            print("  3. View colored roof predictions!")
        else:
            print("  ⚠️  No test images found")
            print("  Add some .png/.jpg files to test inference")

    else:
        print("❌ NO MODELS FOUND - Need to Train First!")
        print("\n🚀 TRAIN A MODEL:")
        print("  1. Open: with_3_classes_improve_from_personal_cloudaV2.ipynb")
        print("  2. Add this at the bottom:")
        print("     if __name__ == '__main__':")
        print("         model, history = quick_fix_main()")
        print("  3. Run the training notebook")
        print("  4. Then use model_inference_visualization.ipynb")

    print("\n💡 TIPS:")
    print("  • Training takes time but fixes the batch size issue")
    print("  • Inference shows colored tile predictions")
    print("  • All paths are now correctly configured!")

def main():
    """Main test function"""
    print("🏠 ROOF TEXTURE CLASSIFICATION - SETUP TEST")
    print("="*60)

    # Test paths
    paths_ok = test_paths()

    # Find available files
    found_images = find_images()
    found_models = find_models()

    # Check notebooks
    check_notebooks()

    # Provide guidance
    provide_next_steps(found_models, found_images, paths_ok)

    print("\n" + "="*60)
    print("✅ Setup test complete!")

if __name__ == "__main__":
    main()
