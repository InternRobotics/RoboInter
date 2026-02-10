import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import tempfile
import time

# ==================== Load from Config ====================
from config import VIDEO_ROOT, ANNOTATIONS

# ==================== Color Configuration ====================
COLORS = {
    'object_box': (66, 133, 244, 100),       # Google Blue
    'placement_proposal': (52, 168, 83, 100), # Google Green
    'gripper_box': (251, 188, 5, 120),        # Google Yellow
    'affordance_box': (234, 67, 53, 100),     # Google Red
    'trace': (156, 39, 176, 200),             # Purple
    'contact_point': (255, 87, 34, 255),      # Deep Orange
    'segmentation': (0, 188, 212, 80),        # Cyan
    'grasp_pose': (76, 175, 80, 200),         # Green
}

COLORS_RGB = {
    'object_box': (66, 133, 244),
    'placement_proposal': (52, 168, 83),
    'gripper_box': (251, 188, 5),
    'affordance_box': (234, 67, 53),
    'trace': (156, 39, 176),
    'contact_point': (255, 87, 34),
    'segmentation': (0, 188, 212),
    'grasp_pose': (76, 175, 80),
}


# ==================== Utility Functions ====================
def get_video_list():
    """Get all video names"""
    return list(ANNOTATIONS.keys())


def load_video_frames(video_name):
    """Load all frames of a video"""
    video_path = os.path.join(VIDEO_ROOT, f"{video_name}.mp4")

    if not os.path.exists(video_path):
        # If video does not exist, generate placeholder frames
        # Determine frame count based on annotation data
        if video_name in ANNOTATIONS:
            max_frame = max(ANNOTATIONS[video_name].keys()) + 1
        else:
            max_frame = 100

        # print(f"Warning: Video {video_name}.mp4 not found, generating {max_frame} placeholder frames")

        frames = []
        for i in range(max_frame):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = (50, 50, 60)  # Dark gray background
            cv2.putText(img, f"Frame {i}", (250, 240), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (200, 200, 200), 2)
            cv2.putText(img, f"Video: {video_name}", (150, 280), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (150, 150, 150), 1)
            cv2.putText(img, "(Video file not found)", (180, 320), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (150, 100, 100), 1)
            frames.append(img)
        return frames

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # print(f"Loaded {len(frames)} frames from {video_name}.mp4")
    return frames


def get_annotation_for_frame(video_name, frame_idx):
    """Get annotation for a specific frame"""
    if video_name not in ANNOTATIONS:
        return None
    video_annot = ANNOTATIONS[video_name]

    # Find annotation for the corresponding frame (annotations might not exist for every frame)
    if frame_idx in video_annot:
        return video_annot[frame_idx]

    # Find the nearest annotated frame
    keys = sorted(video_annot.keys())
    for k in reversed(keys):
        if k <= frame_idx:
            return video_annot[k]
    return video_annot.get(keys[0], None) if keys else None



def draw_box_with_fill(img, box, color_rgba, label=None, border_width=3):
    """Draw a box with fill"""
    if box is None or len(box) != 2:
        return img

    try:
        x1, y1 = int(box[0][0]), int(box[0][1])
        x2, y2 = int(box[1][0]), int(box[1][1])

        # Ensure coordinates are valid
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return img
        if x1 >= img.shape[1] or x2 >= img.shape[1] or y1 >= img.shape[0] or y2 >= img.shape[0]:
            return img
        if x1 >= x2 or y1 >= y2:
            return img

        img_pil = Image.fromarray(img.copy())
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Fill
        draw.rectangle([x1, y1, x2, y2], fill=color_rgba)

        # Border
        border_color = (color_rgba[0], color_rgba[1], color_rgba[2], 255)
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)

        # Label
        if label:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2],
                          fill=(color_rgba[0], color_rgba[1], color_rgba[2], 200))
            draw.text((x1, y1 - 25), label, fill=(255, 255, 255, 255), font=font)

        img_pil = img_pil.convert('RGBA')
        img_pil = Image.alpha_composite(img_pil, overlay)
        return np.array(img_pil.convert('RGB'))

    except (IndexError, ValueError, TypeError) as e:
        print(f"Error drawing box: {e}")
        return img


def draw_trace(img, trace, color_rgba):
    """Draw trace (optical flow effect)"""
    if trace is None or len(trace) < 2:
        return img

    try:
        img_pil = Image.fromarray(img.copy())
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
    except Exception as e:
        print(f"Error initializing trace drawing: {e}")
        return img

    # Draw gradient trace lines
    for i in range(len(trace) - 1):
        # Alpha gradient
        alpha = int(255 * (i + 1) / len(trace))
        color = (color_rgba[0], color_rgba[1], color_rgba[2], alpha)

        x1, y1 = int(trace[i][0]), int(trace[i][1])
        x2, y2 = int(trace[i + 1][0]), int(trace[i + 1][1])

        # Width gradient
        width = max(1, int(5 * (i + 1) / len(trace)))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    # Draw trace points
    for i, point in enumerate(trace):
        alpha = int(255 * (i + 1) / len(trace))
        color = (color_rgba[0], color_rgba[1], color_rgba[2], alpha)
        x, y = int(point[0]), int(point[1])
        radius = max(2, int(6 * (i + 1) / len(trace)))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    # Start point marker (larger circle)
    if len(trace) > 0:
        x, y = int(trace[0][0]), int(trace[0][1])
        draw.ellipse([x - 8, y - 8, x + 8, y + 8],
                    fill=(255, 255, 255, 200),
                    outline=(color_rgba[0], color_rgba[1], color_rgba[2], 255),
                    width=3)

    # End point arrow
    if len(trace) >= 2:
        x1, y1 = trace[-2]
        x2, y2 = trace[-1]
        draw_arrow(draw, x1, y1, x2, y2, color_rgba)

    img_pil = img_pil.convert('RGBA')
    img_pil = Image.alpha_composite(img_pil, overlay)
    return np.array(img_pil.convert('RGB'))


def draw_arrow(draw, x1, y1, x2, y2, color_rgba, arrow_size=12):
    """Draw arrow"""
    import math
    angle = math.atan2(y2 - y1, x2 - x1)

    # Points on both sides of the arrow
    arrow_angle = math.pi / 6
    ax1 = x2 - arrow_size * math.cos(angle - arrow_angle)
    ay1 = y2 - arrow_size * math.sin(angle - arrow_angle)
    ax2 = x2 - arrow_size * math.cos(angle + arrow_angle)
    ay2 = y2 - arrow_size * math.sin(angle + arrow_angle)

    color = (color_rgba[0], color_rgba[1], color_rgba[2], 255)
    draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color)


def draw_contact_points(img, points, color_rgba):
    """Draw contact points"""
    if points is None:
        return img

    img_pil = Image.fromarray(img.copy())
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = int(point[0]), int(point[1])

            # Outer ring
            draw.ellipse([x - 15, y - 15, x + 15, y + 15],
                        fill=(color_rgba[0], color_rgba[1], color_rgba[2], 80))
            # Middle ring
            draw.ellipse([x - 10, y - 10, x + 10, y + 10],
                        fill=(color_rgba[0], color_rgba[1], color_rgba[2], 150))
            # Inner ring
            draw.ellipse([x - 5, y - 5, x + 5, y + 5],
                        fill=(255, 255, 255, 255))
            # Center point
            draw.ellipse([x - 2, y - 2, x + 2, y + 2],
                        fill=(color_rgba[0], color_rgba[1], color_rgba[2], 255))

    img_pil = img_pil.convert('RGBA')
    img_pil = Image.alpha_composite(img_pil, overlay)
    return np.array(img_pil.convert('RGB'))


def draw_segmentation(img, seg_mask, color_rgba):
    """Draw segmentation mask (translucent + edge)"""
    if seg_mask is None:
        return img

    img_pil = Image.fromarray(img.copy())
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))

    # If seg_mask is a numpy array
    if isinstance(seg_mask, np.ndarray):
        mask = seg_mask.astype(np.uint8)

        # Create fill mask
        fill_color = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        fill_color[mask > 0] = color_rgba

        overlay = Image.fromarray(fill_color, 'RGBA')

        # Draw edges
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        cv2.drawContours(edge_img, contours, -1,
                        (color_rgba[0], color_rgba[1], color_rgba[2], 255), 2)
        edge_overlay = Image.fromarray(edge_img, 'RGBA')

        img_pil = img_pil.convert('RGBA')
        img_pil = Image.alpha_composite(img_pil, overlay)
        img_pil = Image.alpha_composite(img_pil, edge_overlay)

    return np.array(img_pil.convert('RGB'))


def draw_grasp_pose(img, points, color_rgba):
    """Draw grasp pose (6 keypoints: center, approach, left_top, right_top, left_bottom, right_bottom)"""
    if points is None or len(points) < 6:
        return img

    import math

    img_pil = Image.fromarray(img.copy())
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pts = [(int(p[0]), int(p[1])) for p in points]
    center, approach, lt, rt, lb, rb = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

    r, g, b = color_rgba[0], color_rgba[1], color_rgba[2]

    # 1) Draw gripper closed area (translucent fill rectangle LT→RT→RB→LB)
    draw.polygon([lt, rt, rb, lb], fill=(r, g, b, 50))

    # 2) Draw two grippers (thick lines representing fingers)
    draw.line([lt, lb], fill=(r, g, b, 255), width=5)
    draw.line([rt, rb], fill=(r, g, b, 255), width=5)

    # 3) Draw top connecting line (gripper base) and bottom connecting line (gripper closure)
    draw.line([lt, rt], fill=(r, g, b, 180), width=2)
    draw.line([lb, rb], fill=(r, g, b, 180), width=2)

    # 4) Draw grasp direction arrow (approach → center)
    arrow_color = (255, 220, 50, 255)
    draw.line([approach, center], fill=arrow_color, width=3)
    angle = math.atan2(center[1] - approach[1], center[0] - approach[0])
    arrow_size = 12
    arrow_angle = math.pi / 6
    ax1 = int(center[0] - arrow_size * math.cos(angle - arrow_angle))
    ay1 = int(center[1] - arrow_size * math.sin(angle - arrow_angle))
    ax2 = int(center[0] - arrow_size * math.cos(angle + arrow_angle))
    ay2 = int(center[1] - arrow_size * math.sin(angle + arrow_angle))
    draw.polygon([center, (ax1, ay1), (ax2, ay2)], fill=arrow_color)

    # 5) Draw center point
    cr = 5
    draw.ellipse([center[0]-cr, center[1]-cr, center[0]+cr, center[1]+cr],
                 fill=(255, 255, 255, 230), outline=(r, g, b, 255), width=2)

    img_pil = img_pil.convert('RGBA')
    img_pil = Image.alpha_composite(img_pil, overlay)
    return np.array(img_pil.convert('RGB'))


# ==================== Main Rendering Function ====================
def render_frame(video_name, frame_idx, display_mode, frames_cache):
    """Render specific frame and its annotations"""
    if frames_cache is None or len(frames_cache) == 0:
        return None, "No video loaded"

    frame_idx = int(frame_idx)
    if frame_idx >= len(frames_cache):
        frame_idx = len(frames_cache) - 1

    img = frames_cache[frame_idx].copy()
    annot = get_annotation_for_frame(video_name, frame_idx)

    text_info = ""

    if annot is None:
        return img, "No annotation for this frame"

    if display_mode == "original":
        text_info = "Original video"

    elif display_mode == "instruction_add":
        text_info = f"Instruction: {annot.get('instruction_add', 'N/A')}"

    elif display_mode == "substask":
        text_info = f"Subtask: {annot.get('substask', 'N/A')}"

    elif display_mode == "primitive_skill":
        text_info = f"Primitive Skill: {annot.get('primitive_skill', 'N/A')}"

    elif display_mode == "object_box":
        seg = annot.get('segmentation')
        if seg is not None and isinstance(seg, np.ndarray):
            # segmentation mask might be 180x320, need to resize to frame dimensions
            if seg.shape[:2] != img.shape[:2]:
                seg = cv2.resize(seg.astype(np.uint8), (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            img = draw_segmentation(img, seg, COLORS['segmentation'])
            text_info = f"Object Segmentation: mask applied"
        else:
            text_info = "Object Segmentation: No data"

    elif display_mode == "placement_proposal":
        box = annot.get('placement_proposal')
        if box:
            img = draw_box_with_fill(img, box, COLORS['placement_proposal'], "Placement")
            text_info = f"Placement Proposal: {box}"
        else:
            text_info = "Placement Proposal: No data"

    elif display_mode == "trace":
        trace = annot.get('trace')
        if trace:
            img = draw_trace(img, trace, COLORS['trace'])
            text_info = f"Trace: {len(trace)} points"
        else:
            text_info = "Trace: No data"

    elif display_mode == "gripper_box":
        box = annot.get('gripper_box')
        if box:
            img = draw_box_with_fill(img, box, COLORS['gripper_box'], "Gripper")
            text_info = f"Gripper Box: {box}"
        else:
            text_info = "Gripper Box: No data"

    elif display_mode == "contact_frame":
        cf = annot.get('contact_frame')
        if cf is not None:
            text_info = f"Contact Frame: {cf} (Click 'Go to Contact Frame' to jump)"
        else:
            text_info = "Contact Frame: No data"

    elif display_mode == "contact_points":
        points = annot.get('contact_points')
        if points:
            img = draw_contact_points(img, points, COLORS['contact_point'])
            text_info = f"Contact Points: {points}"
        else:
            text_info = "Contact Points: No data"

    elif display_mode == "affordance_box":
        box = annot.get('affordance_box')
        if box:
            img = draw_box_with_fill(img, box, COLORS['affordance_box'], "Affordance")
            text_info = f"Affordance Box: {box}"
        else:
            text_info = "Affordance Box: No data"

    elif display_mode == "grasp_pose":
        gp = annot.get('grasp_pose')
        if gp:
            img = draw_grasp_pose(img, gp, COLORS['grasp_pose'])
            text_info = f"Grasp Pose: {len(gp)} keypoints"
        else:
            text_info = "Grasp Pose: No data"

    return img, text_info


# ==================== Gradio Interface ====================
def create_app():
    video_list = get_video_list()

    with gr.Blocks(
        title="Video Annotation Visualizer",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
        css="""
        .annotation-btn {
            min-width: 120px !important;
            padding: 8px 12px !important;
            margin: 3px 0 !important;
            transition: all 0.3s ease !important;
        }
        .annotation-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        }
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
        }
        .frame-slider { margin-top: 10px; }
        .play-controls { display: flex; gap: 10px; align-items: center; }
        .control-panel {
            max-height: 120vh;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 10px;
        }
        .control-panel > * {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        .section-header {
            font-weight: 600;
            margin-top: 12px !important;
            margin-bottom: 6px !important;
            font-size: 14px !important;
        }
        .hint-text {
            font-size: 12px !important;
            color: #666 !important;
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 8px 0 !important;
        }
        .language-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white !important;
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            font-family: monospace;
            line-height: 1.6;
        }
        """
    ) as app:
        # State
        current_video = gr.State(value=video_list[0] if video_list else None)
        frames_cache = gr.State(value=None)
        current_mode = gr.State(value="original")
        current_video_path = gr.State(value=None)  # Current generated video path
        current_language_mode = gr.State(value=None)  # Current language annotation mode

        gr.Markdown("# 🎬 Video Annotation Visualizer")
        gr.Markdown("Visualize various annotations on video frames")

        with gr.Row():
            with gr.Column(scale=3):
                # Use Tabs to switch views
                with gr.Tabs(selected=0) as view_tabs:
                    with gr.Tab("🎬 Video Player", id=0):
                        # Video Player
                        video_player = gr.Video(
                            label="Annotated Video",
                            height=500,
                            show_label=False,
                            autoplay=True
                        )

                        # Video FPS control
                        video_fps = gr.Slider(
                            minimum=5,
                            maximum=30,
                            step=1,
                            value=30,
                            label="Video FPS"
                        )

                    with gr.Tab("🖼️ Frame Viewer", id=1):
                        # Current frame display
                        frame_image = gr.Image(
                            label="Current Frame",
                            height=500,
                            show_label=False
                        )

                        # Frame position control (only shown in Frame Viewer)
                        frame_slider = gr.Slider(
                            minimum=0,
                            maximum=99,
                            step=1,
                            value=0,
                            label="Current Frame Position",
                            elem_classes=["frame-slider"]
                        )

                # Language annotation display area (current frame)
                language_text = gr.Textbox(
                    label="💬 Language Annotation (Current Frame)",
                    interactive=False,
                    lines=3,
                    visible=False,
                    elem_classes=["language-box"]
                )

                # Info display
                info_text = gr.Textbox(
                    label="Annotation Info",
                    interactive=False,
                    elem_classes=["info-box"]
                )

            with gr.Column(scale=1, elem_classes=["control-panel"]):
                # Video Selection
                gr.Markdown("### 📁 Video Selection")
                video_dropdown = gr.Dropdown(
                    choices=video_list,
                    value=video_list[0] if video_list else None,
                    label="Select Video"
                )
                with gr.Row():
                    random_btn = gr.Button("🎲 Random", variant="secondary", size="sm", scale=2)
                    remaining_count = gr.Button(f"📊 {len(video_list)-1}/{len(video_list)}" if video_list else "📊 0/0", variant="secondary", size="sm", scale=2)
                gr.Markdown("### 🎨 Display Mode")

                # Original Video
                original_btn = gr.Button("📹 Original", variant="primary", elem_classes=["annotation-btn"], size="sm")

                # Language Annotation (2x2 layout)
                gr.Markdown("#### 💬 Language")
                with gr.Row():
                    instruction_btn = gr.Button("📝 Instruction", elem_classes=["annotation-btn"], size="sm")
                    subtask_btn = gr.Button("📋 Subtask", elem_classes=["annotation-btn"], size="sm")
                with gr.Row():
                    primitive_btn = gr.Button("⚡ Primitive Skill", elem_classes=["annotation-btn"], size="sm")
                    goto_contact_btn = gr.Button("➡️ Contact Frame", variant="secondary", size="sm")

                gr.Markdown("#### 🎨 Visual")
                with gr.Row():
                    object_box_btn = gr.Button("🎭 Mask", elem_classes=["annotation-btn"], size="sm")
                    placement_btn = gr.Button("📍 Place", elem_classes=["annotation-btn"], size="sm")
                with gr.Row():
                    gripper_btn = gr.Button("🤖 Gripper", elem_classes=["annotation-btn"], size="sm")
                    trace_btn = gr.Button("〰️ Trace", elem_classes=["annotation-btn"], size="sm")

                gr.Markdown("#### 🎯 Contact & Affordance")
                with gr.Row():
                    contact_points_btn = gr.Button("⭕ Contact Points", elem_classes=["annotation-btn"], size="sm")
                    affordance_btn = gr.Button("🔲 Affordance Box", elem_classes=["annotation-btn"], size="sm")
                    grasp_pose_btn = gr.Button("🤏 Grasp Pose", elem_classes=["annotation-btn"], size="sm")

        # ==================== Event Handling ====================

        def load_video(video_name):
            """Load video"""
            frames = load_video_frames(video_name)
            max_frame = len(frames) - 1 if frames else 0
            info = f"Loaded video: {video_name} ({len(frames)} frames)" if frames else "Failed to load video"
            # Get the first frame as initial display
            first_frame = frames[0] if frames else None
            return (
                video_name,
                frames,
                gr.update(maximum=max_frame, value=0),
                first_frame,
                info,
                "original",
                None,  # Clear video path
                None   # Clear language mode
            )

        def random_video(current_video_name):
            """Select a random video"""
            videos = get_video_list()
            if not videos:
                return None
            if len(videos) == 1:
                return videos[0]
            candidates = [v for v in videos if v != current_video_name]
            return random.choice(candidates) if candidates else videos[0]

        def get_remaining_count(current_video_name):
            """Get remaining video count"""
            videos = get_video_list()
            if not videos:
                return "0/0"

            if current_video_name in videos:
                current_idx = videos.index(current_video_name)
                remaining = len(videos) - current_idx - 1
                return f"{remaining}/{len(videos)}"
            else:
                return f"{len(videos)-1}/{len(videos)}"

        def update_current_frame(video_name, frames, frame_idx, mode):
            """Update current frame display based on slider position"""
            if frames is None:
                return None

            frame_idx = int(frame_idx)
            if frame_idx >= len(frames):
                frame_idx = len(frames) - 1

            # Render current frame (apply annotations for current mode)
            img, _ = render_frame(video_name, frame_idx, mode, frames)
            return img

        def get_current_frame_language(video_name, frames, frame_idx, lang_type):
            """Get language annotation for current frame"""
            if frames is None or lang_type is None:
                return "", False

            field_map = {
                "instruction_add": "Instruction",
                "substask": "Subtask",
                "primitive_skill": "Primitive Skill"
            }

            field_name = field_map.get(lang_type, lang_type)
            frame_idx = int(frame_idx)

            annot = get_annotation_for_frame(video_name, frame_idx)
            if annot:
                value = annot.get(lang_type, "N/A")
                if value and value != "N/A":
                    result = f"📝 {field_name} [Frame {frame_idx}]:\n\n{value}"
                    return result, True
                else:
                    return f"No {field_name} annotation for frame {frame_idx}", True

            return f"No annotation data for frame {frame_idx}", True

        def generate_or_play_video(video_name, mode, frames, fps, current_path):
            """Smartly generate/play video"""
            if frames is None or len(frames) == 0:
                return None, "No video loaded", None, "🎬 Generate Video"

            # Check if video already exists and file is still there
            if current_path and os.path.exists(current_path):
                print(f"Using cached video: {current_path}")
                return current_path, f"▶️ Playing cached video ({len(frames)} frames at {fps} FPS)", current_path, "▶️ Play Video"

            # Need to generate a new video
            print(f"Generating video: {video_name}_{mode}.mp4 at {fps} FPS...")

            # Create temporary file
            temp_dir = tempfile.gettempdir()
            video_filename = f"gradio_vis_{video_name}_{mode}_{int(time.time())}.mp4"
            output_path = os.path.join(temp_dir, video_filename)

            # Cleanup old temporary video files (older than 1 hour)
            try:
                import glob
                pattern = os.path.join(temp_dir, "gradio_vis_*.mp4")
                current_time = time.time()
                for old_file in glob.glob(pattern):
                    try:
                        file_age = current_time - os.path.getmtime(old_file)
                        if file_age > 3600:  # 1 hour
                            os.remove(old_file)
                            print(f"  Cleaned up old temp file: {os.path.basename(old_file)}")
                    except Exception as e:
                        pass
            except Exception as e:
                print(f"  Warning: Could not clean up old files: {e}")

            # Get video dimensions
            height, width = frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Render and write each frame
            for i in range(len(frames)):
                img, _ = render_frame(video_name, i, mode, frames)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)

                if (i + 1) % 30 == 0:
                    print(f"  Processed {i + 1}/{len(frames)} frames...")

            out.release()
            print(f"✓ Video saved to: {output_path}")
            print(f"  Note: Temp videos auto-delete after 1 hour")

            return output_path, f"✓ Generated video: {len(frames)} frames at {fps} FPS", output_path, "▶️ Play Video"

        def goto_contact_frame(video_name, frames, fps):
            """Jump to contact frame"""
            if frames is None:
                return 0, "No video loaded"
            annot = get_annotation_for_frame(video_name, 0)
            if annot and annot.get('contact_frame') is not None:
                cf = annot['contact_frame']
                if cf < len(frames):
                    time_sec = cf / fps if fps > 0 else 0
                    info = f"📍 Jumped to Contact Frame: {cf} (at {time_sec:.2f}s with {fps} FPS)"
                    return cf, info
            return 0, "No contact frame available for this video"

        # Define mode buttons list (Visual modes)
        mode_buttons = [
            (original_btn, "original"),
            (object_box_btn, "object_box"),
            (placement_btn, "placement_proposal"),
            (trace_btn, "trace"),
            (gripper_btn, "gripper_box"),
            (contact_points_btn, "contact_points"),
            (affordance_btn, "affordance_box"),
            (grasp_pose_btn, "grasp_pose"),
        ]

        # Define language buttons list
        language_buttons = [
            (instruction_btn, "instruction_add"),
            (subtask_btn, "substask"),
            (primitive_btn, "primitive_skill"),
        ]

        # Combined list of all buttons
        all_buttons = mode_buttons + language_buttons

        def update_button_styles(selected_mode):
            """Update all button styles, highlight selected button"""
            return [
                gr.update(variant="primary" if mode == selected_mode else "secondary")
                for _, mode in all_buttons
            ]

        # ==================== Event Handlers ====================

        # Video selection event (auto-generate original video)
        video_dropdown.change(
            load_video,
            inputs=[video_dropdown],
            outputs=[current_video, frames_cache, frame_slider, frame_image, info_text, current_mode, current_video_path, current_language_mode]
        ).then(
            lambda: tuple(update_button_styles("original")) + (gr.update(visible=False),),
            outputs=[b for b, _ in all_buttons] + [language_text]
        ).then(
            generate_or_play_video,
            inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
            outputs=[video_player, info_text, current_video_path]
        ).then(
            get_remaining_count,
            inputs=[current_video],
            outputs=[remaining_count]
        )

        # Random video button event
        random_btn.click(
            random_video,
            inputs=[current_video],
            outputs=[video_dropdown]
        )

        # Visual mode button events (auto-generate video)
        for btn, mode in mode_buttons:
            def create_visual_handler(m):
                def handler(video_name, frames, frame_idx):
                    # Update current frame display
                    img = update_current_frame(video_name, frames, frame_idx, m)
                    # Switch to Video Player tab (index 0)
                    return (m, None, gr.update(visible=False), None, img, gr.Tabs(selected=0)) + tuple(update_button_styles(m))
                return handler

            btn.click(
                create_visual_handler(mode),
                inputs=[current_video, frames_cache, frame_slider],
                outputs=[current_mode, current_video_path, language_text, current_language_mode, frame_image, view_tabs] + [b for b, _ in all_buttons]
            ).then(
                generate_or_play_video,
                inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
                outputs=[video_player, info_text, current_video_path]
            )

        # Language button events
        for btn, lang_type in language_buttons:
            def create_language_handler(lt):
                def handler(video_name, frames, frame_idx):
                    text, visible = get_current_frame_language(video_name, frames, frame_idx, lt)
                    # Switch to Frame Viewer tab (index 1)
                    return tuple(update_button_styles(lt)) + (gr.update(value=text, visible=visible), lt, gr.Tabs(selected=1))
                return handler

            btn.click(
                create_language_handler(lang_type),
                inputs=[current_video, frames_cache, frame_slider],
                outputs=[b for b, _ in all_buttons] + [language_text, current_language_mode, view_tabs]
            )

        # Frame slider event: update both image and language annotation
        def update_on_frame_change(video_name, frames, frame_idx, mode, lang_mode):
            # Update current frame image
            img = update_current_frame(video_name, frames, frame_idx, mode)

            # Update language annotation (if in language mode)
            if lang_mode is None:
                lang_update = gr.update()
            else:
                text, visible = get_current_frame_language(video_name, frames, frame_idx, lang_mode)
                lang_update = gr.update(value=text, visible=visible)

            return img, lang_update

        frame_slider.change(
            update_on_frame_change,
            inputs=[current_video, frames_cache, frame_slider, current_mode, current_language_mode],
            outputs=[frame_image, language_text]
        )

        # Special button: Jump to contact frame (switch to Frame Viewer, hide Language Annotation)
        def goto_contact_with_tab_switch(video_name, frames, fps):
            frame_idx, info = goto_contact_frame(video_name, frames, fps)
            return frame_idx, info, gr.Tabs(selected=1), gr.update(visible=False), None

        goto_contact_btn.click(
            goto_contact_with_tab_switch,
            inputs=[current_video, frames_cache, video_fps],
            outputs=[frame_slider, info_text, view_tabs, language_text, current_language_mode]
        )

        # Contact Points button: switch mode (switch to Video Player, auto-generate video)
        def switch_to_contact_points_with_frame(video_name, frames, frame_idx):
            img = update_current_frame(video_name, frames, frame_idx, "contact_points")
            return tuple(update_button_styles("contact_points")) + (gr.update(visible=False), None, "contact_points", None, img, gr.Tabs(selected=0))

        contact_points_btn.click(
            switch_to_contact_points_with_frame,
            inputs=[current_video, frames_cache, frame_slider],
            outputs=[b for b, _ in all_buttons] + [language_text, current_language_mode, current_mode, current_video_path, frame_image, view_tabs]
        ).then(
            generate_or_play_video,
            inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
            outputs=[video_player, info_text, current_video_path]
        )

        # Affordance Box button: switch mode (switch to Video Player, auto-generate video)
        def switch_to_affordance_with_frame(video_name, frames, frame_idx):
            img = update_current_frame(video_name, frames, frame_idx, "affordance_box")
            return tuple(update_button_styles("affordance_box")) + (gr.update(visible=False), None, "affordance_box", None, img, gr.Tabs(selected=0))

        affordance_btn.click(
            switch_to_affordance_with_frame,
            inputs=[current_video, frames_cache, frame_slider],
            outputs=[b for b, _ in all_buttons] + [language_text, current_language_mode, current_mode, current_video_path, frame_image, view_tabs]
        ).then(
            generate_or_play_video,
            inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
            outputs=[video_player, info_text, current_video_path]
        )

        # Grasp Pose button: switch mode (logic same as Affordance)
        def switch_to_grasp_pose_with_frame(video_name, frames, frame_idx):
            img = update_current_frame(video_name, frames, frame_idx, "grasp_pose")
            return tuple(update_button_styles("grasp_pose")) + (gr.update(visible=False), None, "grasp_pose", None, img, gr.Tabs(selected=0))

        grasp_pose_btn.click(
            switch_to_grasp_pose_with_frame,
            inputs=[current_video, frames_cache, frame_slider],
            outputs=[b for b, _ in all_buttons] + [language_text, current_language_mode, current_mode, current_video_path, frame_image, view_tabs]
        ).then(
            generate_or_play_video,
            inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
            outputs=[video_player, info_text, current_video_path]
        )

        # Initial load (auto-generate original video)
        app.load(
            load_video,
            inputs=[video_dropdown],
            outputs=[current_video, frames_cache, frame_slider, frame_image, info_text, current_mode, current_video_path, current_language_mode]
        ).then(
            lambda: tuple(update_button_styles("original")) + (gr.update(visible=False),),
            outputs=[b for b, _ in all_buttons] + [language_text]
        ).then(
            generate_or_play_video,
            inputs=[current_video, current_mode, frames_cache, video_fps, current_video_path],
            outputs=[video_player, info_text, current_video_path]
        ).then(
            get_remaining_count,
            inputs=[current_video],
            outputs=[remaining_count]
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()