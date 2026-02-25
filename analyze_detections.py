import json
import os

def normalize_key(key):
    # Remove extension
    base = os.path.splitext(key)[0]
    # Remove masked suffixes if present
    if "_aug0_masked" in base:
        base = base.replace("_aug0_masked", "")
    return base

def calculate_diffs():
    clear_path = r'd:\GitHub-Repos\tiny-face-pytorch\0000058-Novak-Djokovic_clearface\detections.json'
    masked_path = r'd:\GitHub-Repos\tiny-face-pytorch\0000058-Novak-Djokovic_masked\detections.json'

    with open(clear_path, 'r') as f:
        clear_data = json.load(f)
    with open(masked_path, 'r') as f:
        masked_data = json.load(f)

    # Map normalized keys to original keys
    clear_map = {normalize_key(k): k for k in clear_data.keys()}
    masked_map = {normalize_key(k): k for k in masked_data.keys()}

    # Common normalized keys
    common_keys = set(clear_map.keys()) & set(masked_map.keys())

    all_diffs_dict = {}
    magnitude_list = [] # List of (normalized_key, magnitude)

    for norm_key in common_keys:
        clear_key = clear_map[norm_key]
        masked_key = masked_map[norm_key]

        clear_dets = clear_data[clear_key]
        masked_dets = masked_data[masked_key]

        diffs_for_key = []
        # Match detections by index
        for i in range(min(len(clear_dets), len(masked_dets))):
            clear_landmarks = clear_dets[i][-10:]
            masked_landmarks = masked_dets[i][-10:]
            
            # Absolute difference for each landmark coordinate
            diff = [abs(c - m) for c, m in zip(clear_landmarks, masked_landmarks)]
            diffs_for_key.append([round(d, 2) for d in diff])

            # For magnitude ranking, use the first detection
            if i == 0:
                mag = sum(diff)
                magnitude_list.append((norm_key, mag, clear_key))

        all_diffs_dict[norm_key] = diffs_for_key

    # Save the differences dictionary
    with open(r'd:\GitHub-Repos\tiny-face-pytorch\detections_differences.json', 'w') as f:
        json.dump(all_diffs_dict, f, indent=4)

    # Sort to find the top 5 largest differences
    magnitude_list.sort(key=lambda x: x[1], reverse=True)
    top_5 = magnitude_list[:5]

    print("Top 5 keys with largest landmark differences (sum of absolute differences):")
    for norm_key, mag, orig_key in top_5:
        print(f"Key: {orig_key}")
        print(f"  Sum of diffs: {round(mag, 2)}")
        print(f"  Differences: {all_diffs_dict[norm_key][0]}")
        print("-" * 20)

if __name__ == "__main__":
    calculate_diffs()
