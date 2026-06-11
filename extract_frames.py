import os
import sys
import argparse
import cv2

def extract_frames(video_path, start_idx, max_frames, img_ext, jpeg_quality, brightness, contrast):
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        sys.exit(1)

    # 1. Generate clean output folder name from the video filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening stream: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[Error] Failed to parse MJPEG container.")
        sys.exit(1)

    total_stream_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames detected in container: {total_stream_frames}")

    # 2. Advance to the target start index
    if start_idx > 0:
        print(f"  Seeking forward to frame index {start_idx}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
    extracted_count = 0

    print(f"\nExtracting sequences to folder: '{output_dir}/' ...")

    # Ensure format string has a leading dot
    if not img_ext.startswith('.'):
        img_ext = '.' + img_ext

    while True:
        # Check termination constraints
        if max_frames != -1 and extracted_count >= max_frames:
            print(f"Reached requested limit of {max_frames} frames.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video stream.")
            break

        # 3. Generate 6-digit padded file name (e.g., frame_000001.png)
        file_number = extracted_count + 1 + start_idx
        filename = f"frame_{file_number:06d}{img_ext}"
        save_path = os.path.join(output_dir, filename)

        # Brightness adjustment
        frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)

        # Contrast adjustment
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

        # 4. Save frame to disk with corresponding format settings
        if img_ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        else:
            cv2.imwrite(save_path, frame)

        extracted_count += 1

        # Print a progress indicator every 30 frames
        if extracted_count % 30 == 0:
            print(f"  Processed {extracted_count} frames...")

    cap.release()
    print(f"\n[Success] Extraction complete!")
    print(f"  Saved {extracted_count} frames into folder: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from an MJPEG capture file into a dedicated folder structure for MediaPipe/SMPLX."
    )
    
    # Required positional argument
    parser.add_argument("video_path", type=str, help="Path to the input .mjpeg video file.")
    
    # Optional flagged arguments
    parser.add_argument("-s", "--start-idx", type=int, default=0, 
                        help="0-indexed frame number to begin extraction from (default: 0).")
    parser.add_argument("-m", "--max-frames", type=int, default=-1, 
                        help="Maximum number of frames to extract. Set to -1 for all remaining (default: -1).")
    parser.add_argument("-f", "--format", type=str, default="png", choices=["png", "jpg", "jpeg"],
                        help="Output image file format (default: png).")
    parser.add_argument("-q", "--quality", type=int, default=100,
                        help="JPEG quality setting from 1-100. Only applies when format is jpg/jpeg (default: 100).")
    parser.add_argument("-b", "--brightness", type=float, default=1.0,
                        help="Brightness adjustment factor. 1.0 is no adjustment, 0.5 is half brightness, 2.0 is double brightness (default: 1.0).")
    parser.add_argument("-c", "--contrast", type=float, default=1.0,
                        help="Contrast adjustment factor. 1.0 is no adjustment, 0.5 is half contrast, 2.0 is double contrast (default: 1.0).")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        start_idx=args.start_idx,
        max_frames=args.max_frames,
        img_ext=args.format,
        jpeg_quality=args.quality,
        brightness=args.brightness,
        contrast=args.contrast
    )

if __name__ == "__main__":
    main()
