import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    video.release()
    return np.array(frames)

def forward_sam_multi(model_config, model_sam):
    video_path = model_config["video_path"]
    is_video = model_config["is_video"]
    select_frame = model_config["select_frame"]
    direction = model_config["direction"]

    temp_image_list_save_dir = video_path.rsplit(".", 1)[0]
    
    if 'ann_human' in video_path:
        video_path = model_config['origin_video_path']
    video = extract_frames(video_path)
    
    if not is_video:
        video = video[select_frame:select_frame + 1]
    elif direction == "forward":
        video = video[select_frame:]     
    elif direction == "backward":
        video = video[:select_frame+1][::-1]
    
    positive_points_dict = model_config["positive_points"][select_frame]
    negative_points_dict = model_config["negative_points"][select_frame]
    labels_dict = model_config["labels"][select_frame]
    model_sam.set_video_list(video, temp_image_list_save_dir)
    torch.cuda.empty_cache()
    positive_points = [np.array(positive_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    negative_points = [np.array(negative_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    labels = [labels_dict[obj_idx] for obj_idx in positive_points_dict.keys()]

    for i in range(len(positive_points)):
        if len(positive_points[i]) == 0:
            raise ValueError("No positive points in the frame")
        if len(negative_points[i]) != 0:
            positive_points[i] = np.concatenate([positive_points[i], negative_points[i]], axis=0)

    masks_all = model_sam(positive_points, labels, 0, list(positive_points_dict.keys()))
    return masks_all

def forward_sam_multi_stage(model_config, model_sam):
    video_path = model_config["video_path"]
    is_video = model_config["is_video"]
    select_frame = model_config["select_frame"]
    direction = model_config["direction"]

    temp_image_list_save_dir = video_path.rsplit(".", 1)[0]
    
    if 'ann_human' in video_path:
        video_path = model_config['origin_video_path']
    video = extract_frames(video_path)
    
    if not is_video:
        video = video[select_frame:select_frame + 1]
    elif direction == "forward":
        video = video[select_frame:]     
    elif direction == "backward":
        video = video[:select_frame+1][::-1]
    
    positive_points_dict = model_config["positive_points"][select_frame]
    negative_points_dict = model_config["negative_points"][select_frame]
    labels_dict = model_config["labels"][select_frame]

    positive_points = [np.array(positive_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    negative_points = [np.array(negative_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    labels = [labels_dict[obj_idx] for obj_idx in positive_points_dict.keys()]

    for i in range(len(positive_points)):
        if len(positive_points[i]) == 0:
            raise ValueError("No positive points in the frame")
        if len(negative_points[i]) != 0:
            positive_points[i] = np.concatenate([positive_points[i], negative_points[i]], axis=0)
    
    # if length of video is larger than 800, we split the video into N/800 parts
    num_parts = int(len(video) / 800) + 1
    masks_all = np.zeros((len(positive_points), len(video), 1, video[0].shape[0], video[0].shape[1]))
    ind_all = np.arange(len(video))
    for i in range(num_parts):
        video_part = video[i::num_parts]
        ind_part = ind_all[i::num_parts]
        # must contain the first frame
        video_part = np.concatenate([video[:1], video_part], axis=0)
        ind_part = np.concatenate([np.zeros_like(ind_part[:1]), ind_part], axis=0)
        model_sam.set_video_list(video_part, temp_image_list_save_dir)
        masks_all[:, ind_part] = model_sam(positive_points, labels, 0, list(positive_points_dict.keys()))

    return masks_all

def forward_sam(model_config, model_sam):
    video_path = model_config["video_path"]
    is_video = model_config["is_video"]
    select_frame = model_config["select_frame"]
    direction = model_config["direction"]

    temp_image_list_save_dir = video_path.rsplit(".", 1)[0]
    video = extract_frames(video_path)
    if not is_video:
        video = video[select_frame:select_frame + 1]
    elif direction == "forward":
        video = video[select_frame:]     
    elif direction == "backward":
        video = video[:select_frame+1][::-1]
    
    positive_points_dict = model_config["positive_points"]
    negative_points_dict = model_config["negative_points"]
    labels_dict = model_config["labels"]
    
    model_sam.set_video_list(video, temp_image_list_save_dir)
    positive_points = [np.array(positive_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    negative_points = [np.array(negative_points_dict[obj_idx]) for obj_idx in positive_points_dict.keys()]
    labels = [labels_dict[obj_idx] for obj_idx in positive_points_dict.keys()]

    for i in range(len(positive_points)):
        if len(negative_points[i]) != 0:
            positive_points[i] = np.concatenate([positive_points[i], negative_points[i]], axis=0)
    masks_all = model_sam(positive_points, labels, 0, list(positive_points_dict.keys()))
    
    return masks_all

def bidirectional_sam(model_config, model_sam):
    model_config["direction"] = "forward"
    masks_forward = forward_sam(model_config, model_sam)
    if model_config['select_frame'] == 0:
        return masks_forward
    model_config["direction"] = "backward"
    masks_backward = forward_sam(model_config, model_sam)
    
    # mask_image = mask_images_backward[::-1][:-1] + mask_images_forward
    masks = []
    for obj_id in range(len(masks_forward)):
        masks.append(np.concatenate([masks_backward[obj_id][::-1][:-1], masks_forward[obj_id]], axis=0))
    
    return masks

def bidirectional_sam_multi(model_config, model_sam):
    model_config["direction"] = "forward"
    masks_forward = forward_sam_multi(model_config, model_sam)
    if model_config['select_frame'] == 0:
        return masks_forward
    model_config["direction"] = "backward"
    masks_backward = forward_sam_multi(model_config, model_sam)
    
    masks = []
    for obj_id in range(len(masks_forward)):
        masks.append(np.concatenate([masks_backward[obj_id][::-1][:-1], masks_forward[obj_id]], axis=0))
    
    return masks

def bidirectional_sam_multi_mstage(model_config, model_sam):
    model_config["direction"] = "forward"
    masks_forward = forward_sam_multi_stage(model_config, model_sam)
    if model_config['select_frame'] == 0:
        return masks_forward
    model_sam.reset_predictor()
    model_config["direction"] = "backward"
    masks_backward = forward_sam_multi_stage(model_config, model_sam)
    
    masks = []
    for obj_id in range(len(masks_forward)):
        masks.append(np.concatenate([masks_backward[obj_id][::-1][:-1], masks_forward[obj_id]], axis=0))
    
    return np.array(masks, dtype=np.bool_)
      
def predict_sam_video(model_config, model_sam, save_path, time, combined_mask=False):

    if model_config["direction"] == "bidirection":
        masks = bidirectional_sam(copy.deepcopy(model_config), model_sam)
    else:
        masks = forward_sam(copy.deepcopy(model_config), model_sam)

    masks = np.array(masks).astype(np.bool_)
    if combined_mask:
        assert time > 0, "time should be larger than 0 when combined_mask is True"
        select_frame = model_config["select_frame"]
        pre_mask = np.load(save_path.replace(f"/{str(time)}/", f"/{str(time-1)}/"))["masks"]
        if model_config["direction"] == "bidirection":
            new_mask = masks
        elif model_config["direction"] == "forward":
            new_mask = np.zeros_like(pre_mask).astype(np.bool_)
            new_mask[:, select_frame:] = masks
            new_mask[:, :select_frame] = pre_mask[:, :select_frame]
        elif model_config["direction"] == "backward":
            new_mask = np.zeros_like(pre_mask).astype(np.bool_)
            new_mask[:, :select_frame+1] = masks[:, ::-1]
            new_mask[:, select_frame+1:] = pre_mask[:, select_frame+1:]
        masks = new_mask
    
    np.savez_compressed(save_path, masks=masks)
    print(f"Finish processing {model_config['video_path']}")
    
    torch.cuda.empty_cache()
    return masks

def predict_sam_video_multiframe(model_config, model_sam, save_path, time, combined_mask=False, multi_stage=False):
    select_frames = model_config["select_frames"]
    if len(select_frames) > 1:
        assert model_config["direction"] == "bidirection", "while select_frames larger than 1, direction must be bidirection"
    if model_config["direction"] == "bidirection":
        all_masks = []
        for select_frame in select_frames:
            model_config["select_frame"] = select_frame
            if multi_stage:
                masks = bidirectional_sam_multi_mstage(copy.deepcopy(model_config), model_sam)
            else:
                masks = bidirectional_sam_multi(copy.deepcopy(model_config), model_sam)
            all_masks.append(masks)
        masks = combine_masks(all_masks)
    else:
        assert len(select_frames) == 1, "while select_frames larger than 1, direction must be bidirection"
        model_config["select_frame"] = select_frames[0]
        if multi_stage:
            masks = forward_sam_multi_stage(copy.deepcopy(model_config), model_sam)
        else:
            masks = forward_sam_multi(copy.deepcopy(model_config), model_sam)

    # masks = np.array(masks).astype(np.bool_)
    if combined_mask:
        assert time > 0, "time should be larger than 0 when combined_mask is True"
        select_frame = model_config["select_frame"]
        pre_mask = np.load(save_path.replace(f"/{str(time)}/", f"/{str(time-1)}/"))["masks"]
        if model_config["direction"] == "bidirection":
            new_mask = masks
        elif model_config["direction"] == "forward":
            new_mask = np.zeros_like(pre_mask).astype(np.bool_)
            new_mask[:, select_frame:] = masks
            new_mask[:, :select_frame] = pre_mask[:, :select_frame]
        elif model_config["direction"] == "backward":
            new_mask = np.zeros_like(pre_mask).astype(np.bool_)
            new_mask[:, :select_frame+1] = masks[:, ::-1]
            new_mask[:, select_frame+1:] = pre_mask[:, select_frame+1:]
        masks = new_mask
    
    np.savez_compressed(save_path, masks=masks)
    print(f"Finish processing {model_config['video_path']} save to {save_path}")
    
    torch.cuda.empty_cache()
    return masks

def combine_masks(masks_list, t=10):
    mask = masks_list[:][0]
    frame_mask_list = []
    
    for select_frame_id in range(len(masks_list)):
        tmp_list = []
        for obj_id in range(len(masks_list[0])):
            frame_mask = masks_list[select_frame_id][obj_id].sum(-1).sum(-1).sum(-1)
            tmp_list.append(frame_mask)
        frame_mask_list.append(tmp_list)

    for select_frame_id in range(1, len(masks_list)):
        for obj_id in range(len(masks_list[0])):
            mask_or = np.logical_or(mask[obj_id], masks_list[select_frame_id][obj_id])
            mask_and = np.logical_and(mask[obj_id], masks_list[select_frame_id][obj_id])
            less_frame_mask = frame_mask_list[select_frame_id][obj_id] < frame_mask_list[select_frame_id-1][obj_id]
            mask_less = np.where(less_frame_mask[:, np.newaxis, np.newaxis, np.newaxis], masks_list[select_frame_id][obj_id], mask[obj_id])
            mask_and_is_zero_mask = mask_and.sum(-1).sum(-1).sum(-1) < t
            mask_and = np.where(mask_and_is_zero_mask[:, np.newaxis, np.newaxis, np.newaxis], mask_less, mask_and)
            frame_mask = np.logical_and(frame_mask_list[select_frame_id][obj_id] > t, frame_mask_list[select_frame_id-1][obj_id] > t)
            mask[obj_id] = np.where(frame_mask[:, np.newaxis, np.newaxis, np.newaxis], mask_and, mask_or)
    
    return mask

def get_sam_mask_on_image_forward_mutli(model_config, masks_list, video):
    select_frames = model_config["select_frames"]
    positive_points_dict = model_config["positive_points"]
    negative_points_dict = model_config["negative_points"]
    mask_image, width, height = synthesis_image_multi(masks_list, video, positive_points_dict, negative_points_dict, select_frames)
    if mask_image is None:
        return None, None, None
    
    return mask_image, width, height

def get_sam_mask_on_image_forward(model_config, masks_list, video):
    select_frame = model_config["select_frame"]
    positive_points_dict = model_config["positive_points"]
    mask_image, width, height = synthesis_image(masks_list, video, positive_points_dict, select_frame)
    if mask_image is None:
        return None, None, None
    
    return mask_image, width, height

def get_sam_mask_on_image_bidirection(model_config, masks_list, video, select_frame):
    model_config["direction"] = "forward"
    mask_images_forward, width, height = get_sam_mask_on_image_forward(model_config, masks_list, video, select_frame)
    if model_config['select_frame'] == 0:
        mask_image = mask_images_forward
    else:
        model_config["direction"] = "backward"
        mask_images_backward , width, height= get_sam_mask_on_image_forward(model_config, masks_list, video, select_frame)
        mask_image = mask_images_backward[:-1] + mask_images_forward
    
    return mask_image, width, height

def synthesis_image_multi(masks_list, video, positive_points_dict, negative_points_dict, select_frames=[], alpha=0.5):
    if len(select_frames) == 0:
        return None, None, None
    
    obj_ids = list(positive_points_dict[select_frames[0]].keys())
    cmap = plt.get_cmap("tab10")
    colors = [np.array([*cmap(int(i))[:3]]) for i in obj_ids]
    try:
        mask_image = [torch.tensor(masks_list[ii]).permute(2, 3, 1, 0).numpy() * (
            colors[ii].reshape(1, 1, -1)[:, :, :, None]
        ) for ii in range(len(obj_ids))]
    except:
        import ipdb; ipdb.set_trace()
        mask_image = [torch.zeros_like(video).permute(2, 3, 1, 0).numpy() for i in range(len(obj_ids))]
    
    mask_image = [
        (torch.tensor(mask_image[i]).permute(3, 0, 1, 2) * 255)
        .numpy()
        .astype(np.uint8) for i in range(len(mask_image))
    ]
    
    mask_image = [((1-alpha) * video + alpha * mask_image[i]).astype(np.uint8) for i in range(len(mask_image))]
    mix_image_list = []

    # add mask to video
    width, height = mask_image[0][0].shape[1], mask_image[0][0].shape[0]
    text_scale = width / 800
    assert video.shape[0] == mask_image[0].shape[0], f"video shape: {video.shape[0]}, mask shape: {mask_image[0].shape[0]}"
    for i in range(video.shape[0]):      
        for obj_id in range(len(obj_ids)):
            mix_mask = masks_list[obj_id][i][0][:, :, None].repeat(3, axis=2)
            mix_image = np.where(mix_mask, mask_image[obj_id][i], video[i]) if obj_id == 0 else np.where(mix_mask, mask_image[obj_id][i], mix_image)
            # write number on the mask in the image by cv2
            loc = np.where(mix_mask[:,:,0])
            if len(loc[0]) == 0:
                continue
            loc = (np.mean(loc[0]).astype(int), np.mean(loc[1]).astype(int))
            
            if loc[0] < 10:
                loc = (10, loc[1])
            if loc[1] < 10:
                loc = (loc[0], 10)
            if loc[0] > height - 10:
                loc = (height - 10, loc[1])
            if loc[1] > width - 10:
                loc = (loc[0], width - 10)
            
            cv2.putText(mix_image, str(obj_id+1), (loc[1], loc[0]), cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # also draw point on the mask
            if i in select_frames:
                p_points = positive_points_dict[i][obj_id]
                for point in p_points:
                    # green for positive points
                    cv2.circle(mix_image, (point[0], point[1]), 3, (0, 255, 0), -1)
                n_points = negative_points_dict[i][obj_id]
                for point in n_points:
                    # red for negative points
                    cv2.circle(mix_image, (point[0], point[1]), 3, (0, 0, 255), -1)
        
        mix_image_list.append(mix_image)
    
    return mix_image_list, width, height

def synthesis_image(masks_list, video, positive_points_dict, select_frame, alpha=0.5):
    
    obj_ids = list(positive_points_dict.keys())
    cmap = plt.get_cmap("tab10")
    colors = [np.array([*cmap(int(i))[:3]]) for i in obj_ids]
    mask_image = [torch.tensor(masks_list[i]).permute(2, 3, 1, 0).numpy() * (
        colors[i].reshape(1, 1, -1)[:, :, :, None]
    ) for i in range(len(masks_list))]
    
    mask_image = [
        (torch.tensor(mask_image[i]).permute(3, 0, 1, 2) * 255)
        .numpy()
        .astype(np.uint8) for i in range(len(mask_image))
    ]
    
    mask_image = [((1-alpha) * video + alpha * mask_image[i]).astype(np.uint8) for i in range(len(mask_image))]
    mix_image_list = []

    # add mask to video
    width, height = mask_image[0][0].shape[1], mask_image[0][0].shape[0]
    text_scale = width / 800
    assert video.shape[0] == mask_image[0].shape[0], f"video shape: {video.shape[0]}, mask shape: {mask_image[0].shape[0]}"
    for i in range(video.shape[0]):      
        for obj_id in range(len(masks_list)):
            mix_mask = masks_list[obj_id][i][0][:, :, None].repeat(3, axis=2)
            mix_image = np.where(mix_mask, mask_image[obj_id][i], video[i]) if obj_id == 0 else np.where(mix_mask, mask_image[obj_id][i], mix_image)
            # write number on the mask in the image by cv2
            loc = np.where(mix_mask[:,:,0])
            if len(loc[0]) == 0:
                continue
            loc = (np.mean(loc[0]).astype(int), np.mean(loc[1]).astype(int))
            
            if loc[0] < 10:
                loc = (10, loc[1])
            if loc[1] < 10:
                loc = (loc[0], 10)
            if loc[0] > height - 10:
                loc = (height - 10, loc[1])
            if loc[1] > width - 10:
                loc = (loc[0], width - 10)
            
            cv2.putText(mix_image, str(obj_id+1), (loc[1], loc[0]), cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # also draw point on the mask
            if i == select_frame:
                points = positive_points_dict[obj_id]
                for point in points:
                    cv2.circle(mix_image, (point[0], point[1]), 3, colors[obj_id], -1)
        
        mix_image_list.append(mix_image)
    
    return mix_image_list, width, height
