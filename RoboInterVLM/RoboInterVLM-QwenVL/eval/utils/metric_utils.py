import numpy as np
import json
import re
from nltk.translate.bleu_score import sentence_bleu


############# Format #############

def parse_output_multi(text):
    result = {
        "current_bbox": None,
        "target_bbox":  None,
        "grasp_point":  None,
        "future_traj":  None,
        "action":       None,
    }
    # try JSON
    s, e = text.find("{"), text.rfind("}")
    if s!=-1 and e!=-1:
        try:
            j = json.loads(text[s:e+1])
            for k in result:
                if k in j:
                    result[k] = j[k]
            return result
        except:
            pass
    # fallback regex
    def _arr(key):
        m = re.search(rf'"{key}"\s*:\s*(\[[\s\S]*?\])', text)
        if not m: return None
        try: return json.loads(m.group(1))
        except: return None
    result["current_bbox"] = _arr("current_bbox")
    result["target_bbox"]  = _arr("target_bbox")
    result["grasp_point"]  = _arr("grasp_point")
    result["future_traj"]  = _arr("future_traj") or _arr("action")
    return result

def parse_output_single(text, parse_type='bbox'):
    # try JSON
    s, e = text.find("{"), text.rfind("}")
    if s!=-1 and e!=-1:
        try:
            j = json.loads(text[s:e+1])
            return j[parse_type]
        except:
            pass
    # fallback regex
    def _arr(key):
        m = re.search(rf'"{key}"\s*:\s*(\[[\s\S]*?\])', text)
        if not m: return None
        try: return json.loads(m.group(1))
        except: return None

    res = {}
    res[parse_type] = _arr(parse_type)
    return res

############# Utils #############

def compute_iou(b1,b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    uni = a1+a2-inter
    return inter/uni if uni>0 else 0.0

def euclid(p,q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2)**0.5

def compute_action_acc(pred,true,tol=2.0):
    if not isinstance(pred,list) or not isinstance(true,list): return 0.0
    n_pred,n_true=len(pred),len(true)
    if n_pred==0 or n_true==0: return 0.0
    m=0
    for i in range(min(n_pred,n_true)):
        if euclid(pred[i],true[i])<=tol: m+=1
    return m/ max(n_pred,n_true)

def compute_grasp_acc(pred_gp,true_gp,tol=5.0):
    if true_gp is None: return None
    if not isinstance(pred_gp,list): return 0.0
    return 1.0 if euclid(pred_gp,true_gp)<=tol else 0.0

############# Language #############

def robovqa_process_results(doc, results):
    pred = results.replace("\n", "").lower()
    gt = doc["answer"].replace("\n", "").lower()
    if gt in ['yes', 'no']:
        pred = re.sub(r'\b\w*yes\w*\b', 'yes', pred)
        pred = re.sub(r'\b\w*no\w*\b', 'no', pred)
    score, bleu1, bleu2, bleu3, bleu4 = get_bleu_score(pred, gt)
    return_dict = {
        "score": score, 
        "bleu1": bleu1, 
        "bleu2": bleu2, 
        "bleu3": bleu3, 
        "bleu4": bleu4
    }
    return return_dict

def compute_language_metric(prediction, target):
    if type(prediction) == str:
        prediction = [prediction]
    if type(target) == str:
        target = [target]
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    for pred, tgt in zip(prediction, target):
        candidate = list(pred.split(" "))
        reference = [list(tgt.split(" "))]
        if tgt is not None:
            if len(reference[0]) <= 1:
                bleu1.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu2.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu3.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu4.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
            elif len(reference[0]) == 2:
                bleu1.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu2.append(sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00)))
                bleu3.append(sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00)))
                bleu4.append(sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00)))
            elif len(reference[0]) == 3:
                bleu1.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu2.append(sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00)))
                bleu3.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00)))
                bleu4.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00)))
            else:
                bleu1.append(sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00)))
                bleu2.append(sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00)))
                bleu3.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00)))
                bleu4.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))           
    return {
        'score': (np.array(bleu1) + np.array(bleu2) + np.array(bleu3) + np.array(bleu4)) / 4,
        'bleu1': np.array(bleu1),
        'bleu2': np.array(bleu2),
        'bleu3': np.array(bleu3),
        'bleu4': np.array(bleu4)
    }
    
############# Segmentation #############
def compute_iou_batch(b1, b2):

    b2 = b2.reshape(-1, 4)
    if b2.shape[0] > 1:
        return -1
    try:    
        b1 = b1.reshape(-1, 4)[0:1]
    except:
        return 0.0

    x1 = np.maximum(b1[:, 0], b2[:, 0])
    y1 = np.maximum(b1[:, 1], b2[:, 1])
    x2 = np.minimum(b1[:, 2], b2[:, 2])
    y2 = np.minimum(b1[:, 3], b2[:, 3])

    inter_width = np.maximum(0, x2 - x1)
    inter_height = np.maximum(0, y2 - y1)
    inter = inter_width * inter_height

    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    uni = a1 + a2 - inter
    iou = np.where(uni > 0, inter / uni, 0.0)
    iou = np.where(uni > 0, inter / (uni+1e-6), 0.0)

    return iou.item()

def compute_seg_metric(pred, gt):
    try:
        pred = np.array(pred)
    except:
        return {
            'iou': 0
        }
    if pred.shape == (2, 2):
        pred = pred.reshape(-1, 4)
    
    gt = np.array(gt)

    if len(pred.shape) == 1:
        pred = pred[np.newaxis, :]
    if len(gt.shape) == 1:
        gt = gt[np.newaxis, :]
    iou = compute_iou_batch(pred, gt)
    return {
        "iou": iou
    }

############# Trajectory #############
def discrete_frechet_distance(P, Q):
    # N, 2
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(P[0] - Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), np.linalg.norm(P[i] - Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), np.linalg.norm(P[0] - Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                           np.linalg.norm(P[i] - Q[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return c(n - 1, m - 1)

def hausdorff_distance(P, Q):
    # N, 2
    dist_matrix = np.linalg.norm(P[:, np.newaxis] - Q, axis=2)
    d1 = np.max(np.min(dist_matrix, axis=1))
    d2 = np.max(np.min(dist_matrix, axis=0))
    return max(d1, d2)

def root_mean_square_error(pred, gt):
    # B, N, 2
    if len(pred.shape) == 2:
        pred = pred[np.newaxis, ...]
    if len(gt.shape) == 2:
        gt = gt[np.newaxis, ...]
    return np.sqrt(np.mean((pred - gt) ** 2, axis=1)).mean().item()

def dtw(P, Q):
    n, m = len(P), len(Q)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(P[i - 1] - Q[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]

def compute_trajectory_metric(pred, gt):

    if 'future_traj' in pred:
        pred = pred['future_traj']
    
    if pred is None:
        return {
            "dfd": -1,
            "hd": -1,
            "dtw": -1,
            "rmse": -1
        }

    try:
        pred = np.array(pred)
        gt = np.array(gt)
    except:
        return {
            "dfd": -1,
            "hd": -1,
            "dtw": -1,
            "rmse": -1
        }

    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        return {
            "dfd": -1,
            "hd": -1,
            "dtw": -1,
            "rmse": -1
        }
    
    dfd = discrete_frechet_distance(pred, gt).item()
    hd = hausdorff_distance(pred, gt).item()
    dtw_dist = dtw(pred, gt).item()
    rmse = root_mean_square_error(pred, gt) if pred.shape[0] == gt.shape[0] else 0

    return {
        "dfd": dfd,
        "hd": hd,
        "dtw": dtw_dist,
        "rmse": rmse
    }

############# Grasp Point #############
def compute_contact_point_metric(pred, gt):

    if 'contact_point' in pred:
        pred = pred['contact_point']
    
    if pred is None:
        return {
            'point': 0
        }

    try:
        pred = np.array(pred)
        gt = np.array(gt)
    except:
        return {
            'point': 0
        }

    if gt.shape[-1] != 2 or gt.shape[0] != 2:
        return {
            'point': -1
        }

    if pred.shape[-1] == 2 and pred.shape[0] == 2:
        return {
            'point': root_mean_square_error(pred, gt),
        }
    else:
        return {
            'point': 0
        }