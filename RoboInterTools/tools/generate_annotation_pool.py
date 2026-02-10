"""
Generate Annotation Pool

Generate no_annotation.json and has_annotation.json files for the annotation server.
Supports both user-specific and global annotation pools.
"""

import os
import yaml
import json
import argparse


def load_server_config(config_path="./config/config.yaml"):
    """Load server configuration from yaml file."""
    global CONFIG, ROOT_DIR
    with open(config_path) as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    server_config = CONFIG.get("server", {})
    ROOT_DIR = server_config.get("root_dir", "./data")

    return server_config


def load_user_list(user_list_path, output_dir):
    """
    Load user list from file. If not provided, create a default one with 'root'.

    Args:
        user_list_path: Path to user list txt file (one user per line), or None
        output_dir: Output directory (used to create default user_list.txt)

    Returns:
        List of user names
    """
    if user_list_path is None:
        os.makedirs(output_dir, exist_ok=True)
        default_path = os.path.join(output_dir, 'user_list.txt')
        with open(default_path, 'w') as f:
            f.write('root\n')
        print(f"No user list provided. Created default: {default_path}")
        return ['root']

    with open(user_list_path, 'r') as f:
        users = [line.strip() for line in f if line.strip()]

    if not users:
        raise ValueError(f"User list file is empty: {user_list_path}")

    print(f"Loaded {len(users)} users from {user_list_path}")
    return users


def distribute_videos(video_dict, users, save_path_template=None):
    """
    Distribute videos evenly among users in round-robin fashion.

    Args:
        video_dict: {video_path: anno_path} or {video_path: {anno_path, ...}}
        users: List of user names
        save_path_template: Template for save path (use {video_name} as placeholder)

    Returns:
        user_pools: {user_name: {video_path: {anno_path, save_path}}}
    """
    user_pools = {user: {} for user in users}
    video_list = list(video_dict.items())

    for idx, (video_path, value) in enumerate(video_list):
        user = users[idx % len(users)]

        # Handle different input formats
        if isinstance(value, str):
            entry = {'anno_path': value}
            
        else:
            print(f"Warning: Unknown format for {video_path}, skipping")
            continue

        # Generate save_path if not present
        if 'save_path' not in entry:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if save_path_template:
                entry['save_path'] = save_path_template.format(video_name=video_name)
            else:
                entry['save_path'] = ''

        user_pools[user][video_path] = entry

    return user_pools


def generate_annotation_pool(input_path, output_dir, user_list_path=None,
                             save_path_template_lang="", save_path_template_sam=""):
    """
    Generate annotation pool JSON files.

    Args:
        input_path: Path to input JSON file ({video_path: anno_path})
        output_dir: Directory to save output JSON files
        user_list_path: Path to user list file (one user per line)
        save_path_template_lang: Template for lang mode save paths
        save_path_template_sam: Template for sam mode save paths
    """
    # Load input data
    with open(input_path, 'r') as f:
        video_dict = json.load(f)

    if not isinstance(video_dict, dict):
        raise ValueError(f"Expected dict from input JSON, got {type(video_dict)}")

    print(f"Loaded {len(video_dict)} videos from {input_path}")

    # Load or create user list
    users = load_user_list(user_list_path, output_dir)

    for mode in ['lang', 'sam']:
        # Distribute videos evenly among users
        no_annotation = distribute_videos(video_dict, users, save_path_template=save_path_template_lang if mode == 'lang' else save_path_template_sam)

        # has_annotation starts empty for each user
        has_annotation = {user: {} for user in users}

        # Save output files
        os.makedirs(output_dir, exist_ok=True)
    
        no_anno_path = os.path.join(output_dir, f'no_annotation_{mode}.json')
        has_anno_path = os.path.join(output_dir, f'has_annotation_{mode}.json')
        with open(no_anno_path, 'w') as f:
            json.dump(no_annotation, f, indent=2)
        with open(has_anno_path, 'w') as f:
            json.dump(has_annotation, f, indent=2)

        # Print statistics
        print(f"\nGenerated annotation pool:")
        print(f"  no_annotation:  {no_anno_path}")
        print(f"  has_annotation: {has_anno_path}")
        print(f"\nDistribution ({len(video_dict)} videos -> {len(users)} users):")
        for user in users:
            count = len(no_annotation.get(user, {}))
            print(f"  {user}: {count} pending")

    return


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation pool JSON files",
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Input JSON file mapping video_path to anno_path "
                             "(e.g., asserts/demo_data/video_2_lang_anno.json)")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Output directory for JSON files "
                             "(e.g., asserts/demo_data)")
    parser.add_argument('--user-list', type=str, default=None,  # please provide a user list txt file
                        help="Path to user list txt file (one user per line). "
                             "If not provided, a default user_list.txt with 'root' "
                             "will be created in the output directory.")

    args = parser.parse_args()
    
    config = load_server_config()
    args.save_path_template_lang = config.get('save_path_lang_temp', "")
    args.save_path_template_sam = config.get('save_path_sam_temp', "")

    generate_annotation_pool(
        input_path=args.input,
        output_dir=args.output,
        user_list_path=args.user_list,
        save_path_template_lang=args.save_path_template_lang,
        save_path_template_sam=args.save_path_template_sam,
    )


if __name__ == '__main__':
    main()
