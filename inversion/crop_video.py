from moviepy.editor import VideoFileClip

def crop_row_from_video(video_path, output_path, row_index, num_rows):
    # Load the video
    clip = VideoFileClip(video_path)
    
    # Calculate the height of each row
    row_height = int(clip.h / num_rows)
    
    # Define the y-coordinate of the crop (top and bottom of the selected row)
    y1 = row_index * row_height + 5
    y2 = y1 + row_height
    
    # Crop the video to the selected row
    cropped_clip = clip.crop(y1=y1, y2=y2)
    
    # Save the cropped video
    cropped_clip.write_videofile(output_path, codec="libx264")

# Example usage
video_path = "/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/out/cat_renderings/cat_reconstructions_pti.mp4"  # Path to the original video
output_path = "/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/out/cat_renderings/cat_reconstructions_pti_cropped.mp4"  # Path to save the cropped video
row_index = 7  # Index of the row to crop (0-based index, so row 5 is index 4)
num_rows = 9  # Total number of rows in the video

crop_row_from_video(video_path, output_path, row_index, num_rows)