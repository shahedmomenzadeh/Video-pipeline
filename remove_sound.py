import subprocess
from pathlib import Path

def remove_audio_from_dataset(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Define the video formats you are working with
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    
    # Recursively find all files in the input folder
    for video_file in input_path.rglob('*'):
        if video_file.suffix.lower() in video_extensions:
            
            # Maintain the same subfolder structure in the output directory
            relative_path = video_file.relative_to(input_path)
            out_file = output_path / relative_path
            
            # Create the necessary subdirectories if they don't exist
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # FFmpeg command structure
            command = [
                'ffmpeg',
                '-i', str(video_file), # Input file
                '-c:v', 'copy',        # Copy the video stream without re-encoding
                '-an',                 # Remove the audio stream
                '-y',                  # Overwrite output file if it already exists
                str(out_file)          # Output file
            ]
            
            print(f"Muting: {relative_path}")
            
            try:
                # Run the command silently
                subprocess.run(
                    command, 
                    check=True, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_file.name}. It might be corrupted or in an unsupported format.")

# Set your folder paths here
input_directory = 'clip_dataset'
output_directory = 'clip_dataset_muted'

if __name__ == "__main__":
    print(f"Starting to process videos from '{input_directory}'...")
    remove_audio_from_dataset(input_directory, output_directory)
    print("Finished processing all videos!")
