import os
import torch
import argparse
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import imageio  # For merging videos

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_video_in_batches(video_path, mask_path, checkpoint, grid_size, grid_query_frame, backward_tracking, batch_size):
    # Load the video
    video_frames = read_video_from_path(video_path)
    num_frames = video_frames.shape[0]
    
    # Load the segmentation mask
    segm_mask = np.array(Image.open(mask_path))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    # Initialize the model
    if checkpoint is not None:
        model = CoTrackerPredictor(checkpoint=checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(DEFAULT_DEVICE)
    model.eval()

    # Prepare for visualization
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    processed_frames = []  # To store processed video frames

    # Process the video in batches
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        video_batch = video_frames[start:end]
        video_batch = torch.from_numpy(video_batch).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)

        # Process the batch
        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video_batch,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                # segm_mask=segm_mask  # Uncomment if segmentation mask is used
            )
        
        # Save the results for the current batch
        seq_name = os.path.splitext(os.path.basename(video_path))[0]
        vis.visualize(
            video_batch.cpu(),
            pred_tracks.cpu(),
            pred_visibility.cpu(),
            query_frame=0 if backward_tracking else grid_query_frame,
            filename=f"./saved_videos/{seq_name}_batch_{start // batch_size:03d}.mp4",
        )
        
        # Store processed frames
        processed_frames.append(f"./saved_videos/{seq_name}_batch_{start // batch_size:03d}.mp4")

        # Clear GPU memory
        torch.cuda.empty_cache()

    print("Processing completed.")
    merge_videos(processed_frames, seq_name)

def merge_videos(video_paths, output_name):
    # Merge the processed video segments
    writer = imageio.get_writer(f'./saved_videos/{output_name}_merged.mp4', fps=30)  # Adjust fps as needed
    for path in video_paths:
        video = imageio.get_reader(path)
        for frame in video:
            writer.append_data(frame)
    writer.close()
    print(f"Merged video saved as: {output_name}_merged.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./chunk000.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of frames per batch",
    )

    args = parser.parse_args()
    
    process_video_in_batches(
        args.video_path,
        args.mask_path,
        args.checkpoint,
        args.grid_size,
        args.grid_query_frame,
        args.backward_tracking,
        args.batch_size
    )
