import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops


class GLCMKeyFrameAnalyzer:
    """
    A class for analyzing key frames in videos using GLCM texture features.
    """

    def __init__(self, video_path=None, image_dir=None):
        self.frames = []
        self.gray_frames = []
        self.glcms = []
        self.video_info = {}

        if video_path:
            self.load_from_video(video_path)
        elif image_dir:
            self.load_from_directory(image_dir)

    def load_from_video(self, video_path, max_frames=100):
        """Same as original but stores grayscale frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }

        frame_count = self.video_info['frame_count']
        step = max(1, frame_count // max_frames)

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(rgb_frame)
                self.gray_frames.append(gray_frame)

            count += 1
            if len(self.frames) >= max_frames:
                break

        cap.release()
        print(f"Loaded {len(self.frames)} frames from video")

    def load_from_directory(self, image_dir, max_frames=100):
        """Same as original but with grayscale conversion"""
        image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")

        if len(image_files) > max_frames:
            step = len(image_files) // max_frames
            image_files = [image_files[i] for i in range(0, len(image_files), step)][:max_frames]

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(rgb_frame)
                self.gray_frames.append(gray_frame)

        print(f"Loaded {len(self.frames)} frames from directory")

    def compute_glcm_features(self, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256):
        """
        Compute GLCM features for all frames.
        Args:
            distances: List of pixel pair distances
            angles: List of angles in radians
            levels: Number of gray levels
        """
        if not self.gray_frames:
            raise ValueError("No frames loaded. Load frames first.")

        self.glcms = []
        for gray_frame in self.gray_frames:
            # Reduce gray levels for efficiency (optional)
            gray_frame = (gray_frame // (256 // levels)) * (256 // levels)

            glcm = graycomatrix(gray_frame,
                                distances=distances,
                                angles=angles,
                                levels=levels,
                                symmetric=True,
                                normed=True)
            self.glcms.append(glcm)

    def compute_frame_differences(self):
        """Compute texture differences between consecutive frames using GLCM contrast"""
        if not self.glcms:
            self.compute_glcm_features()

        differences = []
        for i in range(1, len(self.glcms)):
            # Compare GLCM contrast (can use other features like energy, homogeneity)
            contrast_prev = graycoprops(self.glcms[i - 1], 'contrast').mean()
            contrast_curr = graycoprops(self.glcms[i], 'contrast').mean()

            # Normalized absolute difference
            diff = abs(contrast_prev - contrast_curr) / max(contrast_prev, contrast_curr)
            differences.append(diff)

        return differences

    def identify_key_frames(self, threshold=0.3):
        """Identify frames where texture difference exceeds threshold"""
        differences = self.compute_frame_differences()
        key_frames = [i + 1 for i, diff in enumerate(differences) if diff > threshold]
        print(f"Identified {len(key_frames)} key frames using GLCM")
        return {
            'key_frames': key_frames,
            'differences': differences
        }

    def visualize_differences(self, threshold=0.3):
        """Plot GLCM differences with threshold"""
        result = self.identify_key_frames(threshold)
        differences = result['differences']

        plt.figure(figsize=(12, 4))
        plt.plot(differences, 'g-', marker='o', label='GLCM Difference')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('GLCM Texture Differences Between Frames')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('glcm_differences.png')
        plt.close()
        print("GLCM differences visualization saved to 'glcm_differences.png'")

    def display_key_frames(self, threshold=0.3):
        """Display identified key frames"""
        result = self.identify_key_frames(threshold)
        key_frames = result['key_frames']

        if not key_frames:
            print("No key frames identified. Try lowering the threshold.")
            return []

        n = len(key_frames)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        for i, frame_idx in enumerate(key_frames):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(self.frames[frame_idx])
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('glcm_key_frames.png')
        plt.close()
        print(f"GLCM key frames visualization saved to 'glcm_key_frames.png'")
        return key_frames


if __name__ == "__main__":
    video_path = "vid2.mp4"  # Change to your video path

    # GLCM-based analysis
    analyzer = GLCMKeyFrameAnalyzer(video_path=video_path)

    print("\nIdentifying key frames with GLCM (threshold=0.3):")
    analyzer.visualize_differences(threshold=0.3)
    key_frames = analyzer.display_key_frames(threshold=0.3)

    print("\nAnalysis complete!")