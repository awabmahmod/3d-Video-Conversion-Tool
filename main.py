import streamlit as st
import os
import shutil
import uuid
import math
from functions import (
    extract_frames,
    estimate_depth_on_frames,
    create_video_from_frames,
    generate_stereo_3d_frames,
    create_stereo_video,
    generate_anaglyph_frames,
    create_anaglyph_video,
)

st.set_page_config(page_title="🎥 3D Video Generator", layout="centered")
st.title("🎬 3D Video Generator")

st.markdown("Upload a video and choose how you want it processed. You can preview the first 10 seconds if needed.")

# ---------------- Sidebar Fine-Tuning Controls ----------------
st.sidebar.header("🎛 Fine-Tuning Parameters")

nth_frame = st.sidebar.number_input("Extract every Nth frame", min_value=1, value=1)
resize_scale = st.sidebar.slider("Resize Scale", 0.1, 1.0, 1.0)
jpeg_quality = st.sidebar.slider("JPEG Quality", 10, 100, 95)

depth_min = st.sidebar.slider("Depth Min Threshold", 0.0, 1.0, 0.0)
depth_max = st.sidebar.slider("Depth Max Threshold", 0.0, 1.0, 1.0)
max_shift = st.sidebar.slider("Max Stereo Shift", 0, 100, 30)
fps = st.sidebar.selectbox("Output FPS", [24, 30, 60], index=0)

# ---------------- Helper: Estimate Time ----------------
def estimate_time(num_frames, process_type="depth", fps=2):
    if process_type == "depth":
        return math.ceil(num_frames / fps)
    elif process_type == "stereo":
        return math.ceil(num_frames / (fps * 0.8))
    elif process_type == "anaglyph":
        return math.ceil(num_frames / (fps * 0.5))
    return 10

# ---------------- Main Controls ----------------
uploaded_video = st.file_uploader("📤 Upload your video", type=["mp4", "mov", "avi"])
option = st.selectbox(
    "Choose processing type:",
    ["Depth Only", "Stereo 3D (Side-by-Side)", "Red/Cyan Anaglyph 3D"]
)

st.subheader("🎞 Processing Options")
enable_preview = st.checkbox("Enable 10s Preview Before Full Generation", value=False)

preview_btn = None
run_btn = None
generate_full_after_preview = False

if enable_preview:
    preview_btn = st.button("🔍 Generate 10s Preview")
    generate_full_after_preview = st.button("🎬 Generate Full Video After Preview")
else:
    run_btn = st.button("🚀 Run Full Video")

# ---------------- Preview Logic ----------------
if uploaded_video and preview_btn:
    with st.spinner("Generating 10-second preview..."):

        session_id = str(uuid.uuid4())
        base_dir = os.path.join("temp", session_id)
        os.makedirs(base_dir, exist_ok=True)

        input_video_path = os.path.join(base_dir, "input.mp4")
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("📥 Extracting preview frames (first 10s)...")
        frame_paths = extract_frames(
            input_video_path,
            output_dir=os.path.join(base_dir, "frames_preview"),
            nth=nth_frame,
            scale=resize_scale,
            jpeg_quality=jpeg_quality,
            max_duration_sec=10
        )

        st.info("🔮 Estimating preview depth...")
        depth_paths = estimate_depth_on_frames(
            frame_paths,
            output_dir=os.path.join(base_dir, "depth_preview"),
            depth_min=depth_min,
            depth_max=depth_max
        )

        if option == "Depth Only":
            preview_path = os.path.join(base_dir, "preview_depth.mp4")
            create_video_from_frames(depth_paths, output_path=preview_path, fps=fps)

        elif option == "Stereo 3D (Side-by-Side)":
            stereo_paths = generate_stereo_3d_frames(
                frame_paths, depth_paths,
                output_dir=os.path.join(base_dir, "stereo_preview"),
                max_shift=max_shift
            )
            preview_path = os.path.join(base_dir, "preview_stereo.mp4")
            create_stereo_video(stereo_paths, output_path=preview_path, fps=fps)

        elif option == "Red/Cyan Anaglyph 3D":
            stereo_paths = generate_stereo_3d_frames(
                frame_paths, depth_paths,
                output_dir=os.path.join(base_dir, "stereo_preview"),
                max_shift=max_shift
            )
            anaglyph_paths = generate_anaglyph_frames(
                stereo_paths,
                output_dir=os.path.join(base_dir, "anaglyph_preview")
            )
            preview_path = os.path.join(base_dir, "preview_anaglyph.mp4")
            create_anaglyph_video(anaglyph_paths, output_path=preview_path, fps=fps)

    st.success("✅ Preview video generated!")
    st.video(preview_path)

# ---------------- Full Video Generation Logic ----------------
if uploaded_video and (run_btn or generate_full_after_preview):
    with st.spinner("Processing full video..."):

        session_id = str(uuid.uuid4())
        base_dir = os.path.join("temp", session_id)
        os.makedirs(base_dir, exist_ok=True)

        input_video_path = os.path.join(base_dir, "input.mp4")
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("📥 Extracting all frames...")
        frame_paths = extract_frames(
            input_video_path,
            output_dir=os.path.join(base_dir, "frames_full"),
            nth=nth_frame,
            scale=resize_scale,
            jpeg_quality=jpeg_quality,
            max_duration_sec=None
        )
        num_frames = len(frame_paths)
        st.success(f"✅ Extracted {num_frames} frames.")

        st.info("🔮 Estimating depth maps...")
        est_sec = estimate_time(num_frames, process_type="depth", fps=2)
        st.caption(f"⏱ Estimated time: {est_sec} seconds")
        depth_paths = estimate_depth_on_frames(
            frame_paths,
            output_dir=os.path.join(base_dir, "depth_full"),
            depth_min=depth_min,
            depth_max=depth_max
        )
        st.success("✅ Depth maps ready.")

        if option == "Depth Only":
            final_path = os.path.join(base_dir, "final_depth.mp4")
            st.info("🎥 Creating depth video...")
            create_video_from_frames(depth_paths, output_path=final_path, fps=fps)

        elif option == "Stereo 3D (Side-by-Side)":
            st.info("🌀 Generating stereo frames...")
            est_sec = estimate_time(num_frames, process_type="stereo", fps=2)
            st.caption(f"⏱ Estimated time: {est_sec} seconds")
            stereo_paths = generate_stereo_3d_frames(
                frame_paths, depth_paths,
                output_dir=os.path.join(base_dir, "stereo_full"),
                max_shift=max_shift
            )
            final_path = os.path.join(base_dir, "final_stereo.mp4")
            create_stereo_video(stereo_paths, output_path=final_path, fps=fps)

        elif option == "Red/Cyan Anaglyph 3D":
            st.info("🎨 Generating stereo frames for anaglyph...")
            stereo_paths = generate_stereo_3d_frames(
                frame_paths, depth_paths,
                output_dir=os.path.join(base_dir, "stereo_full"),
                max_shift=max_shift
            )
            st.success("✅ Stereo views ready.")

            st.info("👓 Creating red/cyan anaglyph...")
            est_sec = estimate_time(num_frames, process_type="anaglyph", fps=2)
            st.caption(f"⏱ Estimated time: {est_sec} seconds")
            anaglyph_paths = generate_anaglyph_frames(
                stereo_paths,
                output_dir=os.path.join(base_dir, "anaglyph_full")
            )
            final_path = os.path.join(base_dir, "final_anaglyph.mp4")
            create_anaglyph_video(anaglyph_paths, output_path=final_path, fps=fps)

    st.success("✅ Full video ready!")
    st.video(final_path)

    with open(final_path, "rb") as f:
        st.download_button("⬇️ Download Video", data=f, file_name=os.path.basename(final_path), mime="video/mp4")

    if st.checkbox("🧹 Delete temporary files"):
        shutil.rmtree(base_dir)
        st.success("Temporary files deleted.")
