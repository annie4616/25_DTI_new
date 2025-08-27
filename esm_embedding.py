import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import pdb
# pip install fair-esm  (또는 esm==2.x 패키지)
import esm
import pandas as pd

print(torch.cuda.device_count())          # 2
print(torch.cuda.get_device_name(0))      # GeForce RTX 3090
print(torch.cuda.get_device_name(1))      # GeForce RTX 3090
pdb.set_trace()



@torch.no_grad()
def load_esm2_650m(device: str = 'cuda:0'):
    # 33layers

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

@torch.no_grad()
def esm_embedding(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: str = "cuda:0",
    repr_layer: int = 33,
    amp: bool = True,
    max_len: int = None
) -> torch.Tensor:
    # 시퀀스 리스트를 받아 ESM2 per-residue representation을 얻고,
    # padding, BOS, EOS를 제외한 residue에 대해 mean pooling한 [B,D] 텐서를 반환
    # ESM batch 입력 준비
    batch = [("",s) for s in sequences]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    # AMP 옵션 (원하면 켜기) : automatic mixed precision
    # 반정밀도(float16, bfloat16) 연산을 자동으로 켜줄지 말지 제어하는 플래그
    autocast_ctx = torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda")))
    with autocast_ctx:
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # [B, L, D]

    # mask 만들기
    pad_idx = alphabet.padding_idx
    bos_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx

    # 유효토큰: not padding
    valid = tokens != pad_idx
    valid = valid & (tokens != bos_idx) & (tokens != eos_idx)

    # 모든 토큰이 특수 토큰인 경우 mean이 NaN이 될 수 있음(길이가 0)
    # 최소 길이를 1로 맞추는 것
    valid_lens = valid.sum(dim=1).clamp_min(1)

    valid = valid.unsqueeze(-1)  # [B, L, 1]
    reps = reps*valid  # [B, L, D] (masked)
    pooled = reps.sum(dim=1)/valid_lens.unsqueeze(-1) #[B, D]

    return pooled.to("cpu", dtype=torch.float32)  # CPU로 옮기고 float32로 변환

@torch.no_grad()
def build_protein_cache_esm2_650m(
    protein_ids: List[str],
    protein_seqs: List[str],
    out_path: str,
    batch_size: int = 16,
    device: str = "cuda:1",
    amp: bool = False,
    max_len: int = None,
) -> Dict[str, torch.Tensor]:
    """
    ESM2 650M으로 mean pooled 임베딩을 추출하여 {id: [D]} 딕셔너리로 저장.
    """
    assert len(protein_ids) == len(protein_seqs)
    model, alphabet, batch_converter = load_esm2_650m(device=device)

    cache: Dict[str, torch.Tensor] = {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(protein_seqs), batch_size), desc="Embedding (ESM2 650M)"):
        batch_ids  = protein_ids[i:i+batch_size]
        batch_seqs = protein_seqs[i:i+batch_size]

        pooled = esm_embedding(
            batch_seqs,
            model=model,
            alphabet=alphabet,
            batch_converter=batch_converter,
            device=device,
            repr_layer=33,
            amp=amp,
            max_len=max_len
        )  # [B, 1280]

        for pid, vec in zip(batch_ids, pooled):
            cache[pid] = vec.contiguous()  # [1280], float32, cpu

    torch.save(cache, out_path)
    return cache


if __name__ == "__main__":
    df = pd.read_csv("data/kiba_train.csv")

    # 중복된 protein ID가 있을 수 있으니 unique하게 처리
    protein_ids = df["ID2"].astype(str).tolist()
    protein_seqs = df["X2"].astype(str).tolist()

    # 혹시 동일한 protein ID가 여러 번 나오면 중복 제거
    unique_dict = {}
    for pid, seq in zip(protein_ids, protein_seqs):
        if pid not in unique_dict:  # 첫 등장만 저장
            unique_dict[pid] = seq

    protein_ids = list(unique_dict.keys())
    protein_seqs = list(unique_dict.values())

    # ==== 캐시 파일 경로 ====
    cache_path = "cache/protein_embeds_esm2_650m_mean.pt"

    # ==== 캐시 빌드 ====
    build_protein_cache_esm2_650m(
        protein_ids=protein_ids,
        protein_seqs=protein_seqs,
        out_path=cache_path,
        batch_size=4,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        amp=True,  # GPU 사용 시 True 권장
        max_len=2000,  # None이면 최대 길이 제한 없음
    )
    print(f"Saved: {cache_path}")