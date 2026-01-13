import os
from pathlib import Path

# Configuration
BASE_DATA_DIR = Path("/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/enzex/Data/InternData/debug_data_train_video")
OUTPUT_SCRIPT_DIR = Path("slurm/sana_video")
TEMPLATE_PATH = Path("slurm/spatialvid_hq/cs/group_0001.sh")

# Folders to process
TARGET_FOLDERS = [
    "fusionX_720p",
    "UltraVideo_960",
    "mjv",
    "sora",
    "fusionX/shift_2",
    "fusionX/shift_4",
    "fusionX/shift_6",
    "pexels_cut_shot",
    "artlist_filtered_new_cut",
]

def generate_chunked_scripts():
    # Read template
    with open(TEMPLATE_PATH, "r") as f:
        template = f.read()

    OUTPUT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    
    CHUNK_SIZE = 20

    for folder in TARGET_FOLDERS:
        full_folder_path = BASE_DATA_DIR / folder
        
        if not full_folder_path.exists():
            print(f"Warning: Folder {full_folder_path} does not exist. Skipping.")
            continue
            
        # Get all zip files
        zip_files = sorted(list(full_folder_path.glob("*.zip")))
        if not zip_files:
            print(f"No zip files found in {folder}. Skipping.")
            continue
            
        print(f"Processing {folder}: {len(zip_files)} zips")
        
        # Split into chunks
        for i, chunk_start in enumerate(range(0, len(zip_files), CHUNK_SIZE)):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(zip_files))
            current_zips = zip_files[chunk_start:chunk_end]
            
            # Format: folder_part001
            part_suffix = f"part{i+1:03d}"
            
            # clean folder name for filename (replace / with _)
            folder_clean = folder.replace("/", "_")
            
            job_name = f"VIPE:{folder_clean}-{part_suffix}"
            script_filename = f"{folder_clean}_{part_suffix}.sh"
            script_path = OUTPUT_SCRIPT_DIR / script_filename
            
            # Log paths
            log_out = f"logs/vipe_{folder_clean}_{part_suffix}_%A_%a.out"
            log_err = f"logs/vipe_{folder_clean}_{part_suffix}_%A_%a.err"
            
            # Output path for results
            # Structure: $MYHOME/data/sana_video_vipe_results/{folder}/
            # We keep the same output folder for all parts so results accumulate there.
            
            # We need to pass the SPECIFIC list of zip files to the script.
            # RawMP4StreamList now accepts comma-separated paths in base_path.
            
            # Construct the comma-separated string of paths
            zip_paths_str = ",".join([str(z) for z in current_zips])
            
            # Customizing the template
            
            # 1. Job Name
            script_content = template.replace(
                "#SBATCH -J VIPE:SpatialVID-HQ-group_0001",
                f"#SBATCH -J {job_name}"
            )
            
            # 2. Logs
            script_content = script_content.replace(
                "#SBATCH -o logs/vipe_group_0001_%A_%a.out",
                f"#SBATCH -o {log_out}"
            )
            script_content = script_content.replace(
                "#SBATCH -e logs/vipe_group_0001_%A_%a.err",
                f"#SBATCH -e {log_err}"
            )
            
            # 3. Base Video Path
            # We replace it with the comma-separated list.
            # IMPORTANT: The string might be very long. 
            # Bash variables have a limit, but typically ~100KB+. 20 paths should be fine.
            # We assume the template uses "$BASE_VIDEO_PATH" (quoted) which handles spaces if any,
            # but here paths don't have spaces usually.
            
            script_content = script_content.replace(
                'BASE_VIDEO_PATH="$MYHOME/data/SpatialVID-HQ/videos/group_0001/"',
                f'BASE_VIDEO_PATH="{zip_paths_str}"'
            )
            
            # 4. Output Path - Point to the output folder
            new_output_path = f'"$MYHOME/data/sana_video_vipe_results/{folder}/"'
            script_content = script_content.replace(
                'OUTPUT_PATH="$MYHOME/data/SpatialVID-HQ/vipe_results/group_0001/"',
                f'OUTPUT_PATH={new_output_path}'
            )
            
            # 5. Time Limit
            # 20 zips is reasonable for 4 hours.
            # script_content = script_content.replace(
            #    "#SBATCH -t 04:00:00",
            #    "#SBATCH -t 04:00:00"
            # )
            
            # Write script
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)

    print(f"Generated scripts in {OUTPUT_SCRIPT_DIR}")

if __name__ == "__main__":
    generate_chunked_scripts()

