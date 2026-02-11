import re
import json
from pathlib import Path


# rh20t grounding data
rh20t_json_root = "TODO: set your rh20t json root path"  # e.g., "/path/to/RH20T_oneM_json/json_file_final/train/llava_format"
rh20t_image_root = ""
rh20t_vla_contact_box_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_contact_box_qa.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_contact_point_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_contact_point_qa.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_current_box_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_current_box_qa.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_final_box_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_final_box_qa.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_traj_init_point_qa_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_traj_qa.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_traj_qa_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_traj_qa_wo_init_pos.json",
    "data_path": f"{rh20t_image_root}",
}
rh20t_vla_gripper_det_qa_train = {
    "annotation_path": f"{rh20t_json_root}/full_single_multi_contact_obj_gripper_det_qa.json",
    "data_path": f"{rh20t_image_root}",
}

# droid grounding data
droid_json_root = "TODO: set your droid json root path"  # e.g., "/path/to/droid_oneM_json/json_file_final/train/llava_format"
droid_image_root = ""
droid_vla_contact_box_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_contact_box_qa.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_contact_point_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_contact_point_qa.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_current_box_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_current_box_qa.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_final_box_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_final_box_qa.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_traj_init_point_qa_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_traj_qa.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_traj_qa_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_traj_qa_wo_init_pos.json",
    "data_path": f"{droid_image_root}",
}
droid_vla_gripper_det_qa_train = {
    "annotation_path": f"{droid_json_root}/full_single_multi_contact_obj_gripper_det_qa.json",
    "data_path": f"{droid_image_root}",
}


##################################################################################
# REAL Dataset: Understanding
##################################################################################

# rh20t understanding data
rh20t_choice_json_root = "TODO: set your rh20t choice json root path"  # e.g., "/path/to/VQA_choice/split/train"
rh20t_choice_image_root = "TODO: set your rh20t choice image root path"  # e.g., "/path/to/VQA_choice"
rh20t_contact_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/rh20t_contact_decide.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_graspppose_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/grasppose_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_grounding_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/grounding_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_traj_lang_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/trajlang_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_traj_lang_sub_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/trajlang_sub_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_traj_direction_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/traj_direction_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_traj_choice_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/traj_choice_rh20t.json",
    "data_path": f"{rh20t_choice_image_root}",
}
rh20t_traj_direction_choice_with_traj_qa_train = {
    "annotation_path": f"{rh20t_choice_json_root}/traj_direction_choice_rh20t_with_traj.json",
    "data_path": f"{rh20t_choice_image_root}",
}

# droid understanding data
droid_choice_json_root = "TODO: set your droid choice json root path"  # e.g., "/path/to/VQA_choice/split/train"
droid_choice_image_root = "TODO: set your droid choice image root path"  # e.g., "/path/to/VQA_choice"
droid_contact_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/droid_contact_decide.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_grounding_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/grounding_choice_droid.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_traj_lang_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/trajlang_choice_droid.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_traj_lang_sub_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/trajlang_sub_choice_droid.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_traj_direction_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/traj_direction_choice_droid.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_traj_choice_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/traj_choice_droid.json",
    "data_path": f"{droid_choice_image_root}",
}
droid_traj_direction_choice_with_traj_qa_train = {
    "annotation_path": f"{droid_choice_json_root}/traj_direction_choice_droid_with_traj.json",
    "data_path": f"{droid_choice_image_root}",
}

# manipvqa language data
manipvqa_json_root = "TODO: set your manipvqa json root path"  # e.g., "/path/to/VQA_task_planning/split/train"
manipvqa_image_root = "TODO: set your manipvqa image root path"  # e.g., "/path/to/QA_image_v1"
manipvqa_train = {
    "annotation_path": f"{manipvqa_json_root}/all_plan_VQA.json",
    "data_path": manipvqa_image_root,
}


##################################################################################
# Register all datasets to data_dict
##################################################################################



data_dict = {
    # rh20t grounding data
    "rh20t_vla_contact_box_train": rh20t_vla_contact_box_train,
    "rh20t_vla_contact_point_train": rh20t_vla_contact_point_train,
    "rh20t_vla_current_box_train": rh20t_vla_current_box_train,
    "rh20t_vla_final_box_train": rh20t_vla_final_box_train,
    "rh20t_vla_traj_qa_train": rh20t_vla_traj_qa_train,
    "rh20t_vla_traj_init_point_qa_train": rh20t_vla_traj_init_point_qa_train,
    "rh20t_vla_gripper_det_qa_train": rh20t_vla_gripper_det_qa_train,
    # droid grounding data
    "droid_vla_contact_box_train": droid_vla_contact_box_train,
    "droid_vla_contact_point_train": droid_vla_contact_point_train,
    "droid_vla_current_box_train": droid_vla_current_box_train,
    "droid_vla_final_box_train": droid_vla_final_box_train,
    "droid_vla_traj_qa_train": droid_vla_traj_qa_train,
    "droid_vla_traj_init_point_qa_train": droid_vla_traj_init_point_qa_train,
    "droid_vla_gripper_det_qa_train": droid_vla_gripper_det_qa_train,
    # rh20t understanding data
    "rh20t_contact_choice_qa_train": rh20t_contact_choice_qa_train,
    "rh20t_graspppose_choice_qa_train": rh20t_graspppose_choice_qa_train,
    "rh20t_grounding_choice_qa_train": rh20t_grounding_choice_qa_train,
    "rh20t_traj_lang_choice_qa_train": rh20t_traj_lang_choice_qa_train,
    "rh20t_traj_lang_sub_choice_qa_train": rh20t_traj_lang_sub_choice_qa_train,
    "rh20t_traj_direction_choice_qa_train": rh20t_traj_direction_choice_qa_train,
    "rh20t_traj_choice_qa_train": rh20t_traj_choice_qa_train,
    "rh20t_traj_direction_choice_with_traj_qa_train": rh20t_traj_direction_choice_with_traj_qa_train,
    # droid understanding data
    "droid_contact_choice_qa_train": droid_contact_choice_qa_train,
    "droid_grounding_choice_qa_train": droid_grounding_choice_qa_train,
    "droid_traj_lang_choice_qa_train": droid_traj_lang_choice_qa_train,
    "droid_traj_lang_sub_choice_qa_train": droid_traj_lang_sub_choice_qa_train,
    "droid_traj_direction_choice_qa_train": droid_traj_direction_choice_qa_train,
    "droid_traj_choice_qa_train": droid_traj_choice_qa_train,
    "droid_traj_direction_choice_with_traj_qa_train": droid_traj_direction_choice_with_traj_qa_train,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


def count_data(data_dicts):
    for data_name in data_dicts:
        annotation_path = data_dicts[data_name]["annotation_path"]
        if annotation_path.endswith(".jsonl"):
            with open(annotation_path, "r") as f:
                data = f.readlines()
            print(data_name, len(data))
        else:
            with open(annotation_path, "r") as f:
                data = json.load(f)
            print(data_name, len(data))


if __name__ == "__main__":
    count_data(data_dict)