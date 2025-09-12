# prediction.py
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import timm

# ===============================
# Utilities
# ===============================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# โมเดลที่รองรับ (เพิ่มได้ง่าย ๆ)
MODEL_REGISTRY: Dict[str, Dict] = {
    "convnext_tiny.fb_in22k_ft_in1k": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k",
        "img_size": 256,
    },
    "efficientnet_b0.ra_in1k": {
        "timm_name": "efficientnet_b0.ra_in1k",
        "img_size": 256,
    },
    "mobilenetv3_large_100.ra_in1k": {
        "timm_name": "mobilenetv3_large_100.ra_in1k",
        "img_size": 256,
    },
    "ecaresnet50d.miil_in1k": {
        "timm_name": "ecaresnet50d.miil_in1k",  # หรือ "ecaresnet50d"
        "img_size": 256,
    },
    "inception_v3.tf_in1k": {
        "timm_name": "inception_v3.tf_in1k",
        "img_size": 256,
    },
}

def build_model(model_key: str, num_classes: int) -> torch.nn.Module:
    """สร้างสถาปัตยกรรมจาก timm ตามคีย์"""
    cfg = MODEL_REGISTRY[model_key]
    model = timm.create_model(cfg["timm_name"], num_classes=num_classes, pretrained=False)
    return model

def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    รองรับทุกแบบที่เจอบ่อย:
    - torch.save(model)    (โหลดกลับด้วย torch.load แล้วใช้ .state_dict() ได้)
    - torch.save(model.state_dict())
    - checkpoint dict ที่มี key: 'state_dict' / 'model' / อื่นๆ
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # กรณี save ทั้ง model (ไม่ใช่ state_dict)
    if isinstance(obj, torch.nn.Module):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            # เดาว่า obj คือ state_dict ตรง ๆ
            sd = obj
    else:
        # สุดท้าย: พยายามตีความว่าเป็น state_dict
        sd = obj

    # เอา prefix "module." ออกถ้ามี (จาก DataParallel)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)  # ปล่อยหลวมเพื่อความเข้ากัน
    # สามารถ print ใน console ถ้าต้องการตรวจสอบ:
    if len(missing) > 0 or len(unexpected) > 0:
        print("[checkpoint] missing keys:", missing)
        print("[checkpoint] unexpected keys:", unexpected)

def make_transform(img_size: int):
    # ใช้ torchvision แบบ lightweight ภายใน torch: transforms.functional
    from torchvision.transforms.functional import resize, to_tensor, normalize
    def _tfm(pil_img: Image.Image) -> torch.Tensor:
        # resize แบบ bilinear, keep ratio + center crop อย่างง่าย
        img = pil_img.convert("RGB")
        img = resize(img, [img_size, img_size])
        x = to_tensor(img)
        x = normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return x
    return _tfm

# ===============================
# Inference
# ===============================

def predict_probs(
    model: torch.nn.Module,
    image: Image.Image,
    img_size: int,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """คืนค่า probs (1, C)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    tfm = make_transform(img_size)
    x = tfm(image).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

def topk_from_probs(probs: np.ndarray, labels: List[str], k: int = 5):
    """คืนค่ารายการ top-k (label, prob) เรียงจากมากไปน้อย"""
    p = probs[0]
    k = min(k, len(p))
    idx = np.argsort(-p)[:k]
    return [(labels[i], float(p[i])) for i in idx]
