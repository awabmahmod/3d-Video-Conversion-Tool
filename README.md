# 🎥 3D Video Generator

A Streamlit web application for converting regular 2D videos into immersive 3D experiences using deep learning and image processing. Choose from three processing options: **Depth Map**, **Stereo 3D (Side-by-Side)**, and **Red/Cyan Anaglyph 3D**.

---

## 🚀 Features

- ✅ Upload video in `.mp4`, `.mov`, or `.avi` format
- 🧠 Depth estimation using the Intel DPT-Hybrid-MiDaS model
- 🔁 Frame extraction with adjustable quality and resolution
- 👓 Generate:
  - Depth-only videos
  - Side-by-side stereo 3D videos
  - Red/Cyan anaglyph 3D videos
- ⚙️ Customizable processing parameters
- 🔍 Preview the first 10 seconds before running full processing
- 💾 Download final output video

---

## 📸 Processing Modes

1. **Depth Only**
   - Converts frames into grayscale depth maps.
2. **Stereo 3D (Side-by-Side)**
   - Uses depth information to generate stereo pairs.
3. **Red/Cyan Anaglyph 3D**
   - Merges stereo pairs into a viewable format using standard 3D glasses.

---

## 🧰 Technologies Used

- [Streamlit](https://streamlit.io/) for UI
- [Transformers](https://huggingface.co/transformers/) (`Intel/dpt-hybrid-midas`) for depth estimation
- [OpenCV](https://opencv.org/) for frame processing
- [ImageIO](https://imageio.readthedocs.io/) for video creation
- [PyTorch](https://pytorch.org/) for inference

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-video-generator.git
cd 3d-video-generator

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Folder Structure

```
.
├── main.py                 # Streamlit app logic
├── functions.py           # Processing functions
├── requirements.txt       # Python dependencies
├── temp/                  # Temporary folders generated per session
```

---

## 🧮 Parameters

- `Extract every Nth frame` – skip frames to reduce load
- `Resize Scale` – downscale for speed
- `JPEG Quality` – image quality vs. size
- `Depth Min/Max Threshold` – normalize depth maps
- `Max Stereo Shift` – how far views are shifted for stereo
- `Output FPS` – frame rate of final video

---

## 📌 Notes

- The app caches the depth estimation model using `@st.cache_resource`.
- CUDA is used if available for faster processing.
- Output videos can be downloaded after generation.
- Temporary files are cleaned up if the user opts to delete them.

---

## 🤝 Contributing

Feel free to fork, submit issues or PRs.
