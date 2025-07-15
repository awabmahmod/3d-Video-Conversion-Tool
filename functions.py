import os
import cv2
import torch
import imageio
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import streamlit as st

def extract_frames(video_path, output_dir='frames', nth=1, scale=1.0, jpeg_quality=95, max_duration_sec=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    if max_duration_sec:
        max_frames = int(min(frame_count, fps * max_duration_sec))
    else:
        max_frames = frame_count

    frame_paths = []
    idx, saved_idx = 0, 0

    print(f"Extracting up to {max_frames} frames every {nth}th frame (scale: {scale})")

    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % nth == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            frame_path = os.path.join(output_dir, f'frame_{saved_idx:05d}.jpg')
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frame_paths.append(frame_path)
            saved_idx += 1
        idx += 1

    cap.release()
    print(f"✅ Saved {len(frame_paths)} frames to {output_dir}")
    return frame_paths


# Example usage (replace with your video file path)
# frames = extract_frames('your_video.mp4')

#frames = extract_frames(r"C:\Users\COMPUMARTS\Downloads\depth trial\Aayan_Opt2_NEW.mp4")



from PIL import Image

def estimate_depth_on_frames(frame_paths, output_dir='depth_frames', depth_min=0.0, depth_max=1.0):
    from PIL import Image
    from streamlit import cache_resource

    os.makedirs(output_dir, exist_ok=True)

    @cache_resource
    def load_model():
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return processor, model, device

    processor, model, device = load_model()
    model.eval()

    for path in tqdm(frame_paths, desc="Processing depth"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize with thresholds
        dmin = max(depth_min, prediction.min())
        dmax = min(depth_max, prediction.max())
        norm = np.clip((prediction - dmin) / (dmax - dmin + 1e-8), 0, 1)
        depth_8bit = (norm * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, depth_8bit)

    print(f"✅ Depth maps saved to {output_dir}")
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])


def create_video_from_frames(frame_paths, output_path='depth_video.mp4', fps=24):
    if not frame_paths:
        print("No frames to compile.")
        return
    
    # Read the first frame to get the resolution
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    writer = imageio.get_writer(output_path, fps=fps)

    print(f"Creating video at: {output_path}")

    for frame_path in tqdm(frame_paths, desc="Writing video"):
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # convert to 3-channel
        writer.append_data(frame_rgb)

    writer.close()
    print(f"✅ Video saved: {output_path}")


#create_video_from_frames(depth_frame_paths, output_path='depth_video.mp4', fps=24)


def generate_stereo_3d_frames(rgb_frame_paths, depth_frame_paths, output_dir='stereo_frames', max_shift=30):
    os.makedirs(output_dir, exist_ok=True)
    stereo_paths = []

    for rgb_path, depth_path in tqdm(zip(rgb_frame_paths, depth_frame_paths), total=len(rgb_frame_paths), desc="Generating stereo frames"):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        h, w = depth.shape

        depth = depth.astype(np.float32) / 255.0
        shift_map = (1.0 - depth) * max_shift

        left_view = np.zeros_like(rgb)
        right_view = np.zeros_like(rgb)

        for y in range(h):
            for x in range(w):
                dx = int(shift_map[y, x])
                if x - dx >= 0:
                    left_view[y, x] = rgb[y, x - dx]
                if x + dx < w:
                    right_view[y, x] = rgb[y, x + dx]

        stereo_frame = np.concatenate((left_view, right_view), axis=1)
        out_path = os.path.join(output_dir, os.path.basename(rgb_path))
        cv2.imwrite(out_path, stereo_frame)
        stereo_paths.append(out_path)

    return stereo_paths



#stereo_frame_paths = generate_stereo_3d_frames(frames, depth_frame_paths, output_dir='stereo_frames', max_shift=30)

def create_stereo_video(frame_paths, output_path='stereo_video.mp4', fps=24):
    if not frame_paths:
        print("No stereo frames to compile.")
        return

    # Read first frame to get resolution
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    writer = imageio.get_writer(output_path, fps=fps)

    print(f"Creating stereo video at: {output_path}")

    for path in tqdm(frame_paths, desc="Writing stereo video"):
        frame = cv2.imread(path)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # imageio expects RGB

    writer.close()
    print(f"✅ Stereo 3D video saved: {output_path}")


#create_stereo_video(stereo_frame_paths, output_path='stereo_video.mp4', fps=24)


def generate_anaglyph_frames(stereo_frame_paths, output_dir='anaglyph_frames'):
    os.makedirs(output_dir, exist_ok=True)
    anaglyph_paths = []

    for path in tqdm(stereo_frame_paths, desc="Generating anaglyph frames"):
        stereo = cv2.imread(path)
        h, w = stereo.shape[:2]
        mid = w // 2

        left = stereo[:, :mid]
        right = stereo[:, mid:]

        # Create anaglyph: R from left, G+B from right
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 0]       # Red channel
        anaglyph[:, :, 1] = right[:, :, 1]      # Green channel
        anaglyph[:, :, 2] = right[:, :, 2]      # Blue channel

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, anaglyph)
        anaglyph_paths.append(out_path)

    return anaglyph_paths


#anaglyph_frame_paths = generate_anaglyph_frames(stereo_frame_paths)


def create_anaglyph_video(frame_paths, output_path='anaglyph_video.mp4', fps=24):
    if not frame_paths:
        print("No anaglyph frames to compile.")
        return

    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    writer = imageio.get_writer(output_path, fps=fps)

    print(f"Creating anaglyph video at: {output_path}")

    for path in tqdm(frame_paths, desc="Writing anaglyph video"):
        frame = cv2.imread(path)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # imageio wants RGB

    writer.close()
    print(f"✅ Anaglyph 3D video saved: {output_path}")


#create_anaglyph_video(anaglyph_frame_paths, output_path='anaglyph_video.mp4', fps=24)
