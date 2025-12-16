import os
import re

LOG_DIR = 'logs'
SUCCESS_STRING = "Inference finished successfully."

def get_group_id(filename):
    # Pattern: vipe_group_{id}_...
    # Extracts the first number sequence after vipe_group_
    match = re.search(r'vipe_group_(\d+)', filename)
    if match:
        return match.group(1)
    return None

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Directory '{LOG_DIR}' does not exist.")
        return

    # Map group_id to list of files
    groups = {}
    
    try:
        files = os.listdir(LOG_DIR)
    except OSError as e:
        print(f"Error listing directory {LOG_DIR}: {e}")
        return

    for f in files:
        if f.endswith('.out') and f.startswith('vipe_group_'):
            gid = get_group_id(f)
            if gid:
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(os.path.join(LOG_DIR, f))
    
    finished_groups = []
    
    # Check each group
    for gid, file_list in sorted(groups.items()):
        is_finished = False
        finished_file = None
        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read line by line to avoid loading huge files into memory if not needed,
                    # though read() is simpler for small files. 
                    # Given some files are small (4KB) and others larger, reading content is fine 
                    # but line by line is safer for very large logs.
                    for line in f:
                        if SUCCESS_STRING in line:
                            is_finished = True
                            finished_file = file_path
                            break
                if is_finished:
                    break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if is_finished:
            finished_groups.append((gid, finished_file))
            
    print("Finished Group IDs and Files:")
    if not finished_groups:
        print("None found.")
    else:
        for gid, fname in sorted(finished_groups):
            print(f"Group {gid}: {fname}")

if __name__ == "__main__":
    main()

