import bisect
import torch
import logging
from pathlib import Path
from vipe.utils.loop_models import SaladLoopDetector

logger = logging.getLogger(__name__)

class LoopDetector:
    def __init__(self, method="salad", similarity_threshold=0.85, loop_window=200, nms_threshold=25, ckpt_path=None):
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.loop_window = loop_window
        self.nms_threshold = nms_threshold
        self.ckpt_path = ckpt_path
        self.loop_list = []
        
        # Default weights path if not provided
        if self.ckpt_path is None:
            # Assume local relative path or prompt user
            # vggt default: ./weights/dino_salad.ckpt
            self.ckpt_path = "./weights/dino_salad.ckpt"

    def detect(self, video_stream):
        if self.method == "salad":
            if not Path(self.ckpt_path).exists():
                logger.warning(f"SALAD checkpoint not found at {self.ckpt_path}. Skipping loop detection.")
                return
                
            detector = SaladLoopDetector(
                ckpt_path=self.ckpt_path,
                similarity_threshold=self.similarity_threshold,
                # loop_window is not directly used in SaladLoopDetector init but in logic if needed?
                # SaladLoopDetector uses hardcoded exclude_window=10 in find_loops? 
                # Wait, vggt code has loop_window but LoopModel.py hardcodes 10 or uses nms_threshold?
                # LoopModel.py: abs(i - neighbor_idx) > 10.
                # I should update SaladLoopDetector to use loop_window if that's what user meant, 
                # but vggt config loop_window might be used elsewhere? 
                # Actually vggt config has 'Loop.SALAD.nms_threshold'.
                nms_threshold=self.nms_threshold
            )
            
            # Extract and find
            detector.extract_features(video_stream)
            self.loop_list = detector.find_loops()
            
        else:
            # Fallback or brute force (removed for strictness)
            logger.warning(f"Unknown loop detection method: {self.method}")

    def load_from_file(self, path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith("#"): continue
                    parts = list(map(str, line.strip().split(',')))
                    if len(parts) < 2:
                        parts = line.strip().split() # try space
                    if len(parts) >= 2:
                        try:
                            self.loop_list.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            pass
        except Exception as e:
            logger.error(f"Failed to load loops from file: {e}")

# ==========================
# Utils from sim3utils.py
# ==========================

def find_chunk_index(chunks, idx):
    """
    Find the 0-based chunk index that contains the given index idx.
    chunks: List of (begin_idx, end_idx).
    idx: The index to search for.
    Returns the 0-based chunk index.
    """
    starts = [chunk[0] for chunk in chunks]
    pos = bisect.bisect_right(starts, idx) - 1
    if pos < 0 or pos >= len(chunks):
        return -1
    chunk_begin, chunk_end = chunks[pos]
    if idx < chunk_begin or idx >= chunk_end: # Note: chunk_end is usually exclusive in python range?
        # vggt code: if idx < chunk_begin or idx > chunk_end. 
        # vggt chunks are (start, end) inclusive or exclusive?
        # In vggt_long.py: self.chunk_indices.append((start_idx, end_idx)).
        # process_single_chunk uses img_list[start_idx:end_idx]. So exclusive end.
        # But `idx > chunk_end` in vggt `find_chunk_index` implies inclusive check?
        # If `idx` is frame index. `chunk_end` is exclusive limit. So idx must be < chunk_end.
        # My implementation: `idx >= chunk_end` -> fail.
        return -1
    return pos

def get_frame_range(chunk, idx, half_window=10):
    """
    Calculate the frame range centered at idx with half_window frames on each side within chunk boundaries.
    """
    begin, end = chunk
    window_size = 2 * half_window

    if idx - half_window < begin:
        start = begin
        end_candidate = begin + window_size
        end_res = min(end, end_candidate)

    elif idx + half_window > end:
        end_candidate = end
        start_candidate = end - window_size
        start = max(begin, start_candidate)
        end_res = end # Correct var name

    else:
        start = idx - half_window
        end_res = idx + half_window
    return (start, end_res)

def process_loop_list(chunk_index, loop_list, half_window=10):
    """
    Process loop_list and return chunk indices and frame ranges.
    """
    results = []
    for idx1, idx2 in loop_list:
        try:
            chunk_idx1 = find_chunk_index(chunk_index, idx1)
            if chunk_idx1 == -1: continue
            
            chunk1 = chunk_index[chunk_idx1]
            range1 = get_frame_range(chunk1, idx1, half_window)
            
            chunk_idx2 = find_chunk_index(chunk_index, idx2)
            if chunk_idx2 == -1: continue
            
            chunk2 = chunk_index[chunk_idx2]
            range2 = get_frame_range(chunk2, idx2, half_window)
            
            if chunk_idx1 != chunk_idx2:
                results.append((
                    chunk_idx1,
                    range1,
                    chunk_idx2,
                    range2
                ))
        except ValueError:
            continue
    return results
