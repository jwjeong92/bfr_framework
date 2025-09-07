import torch
import numpy as np
from numba import njit

# ----------------------
# [1] Fallback (임의 wbit)용 Numba JIT 함수
# ----------------------

@njit
def _pack_bits_jit(flat: np.ndarray, wbit: int) -> np.ndarray:
    """
    1차원 배열 flat을 wbit비트씩 int32 배열에 '비트 스트림' 형태로 패킹.
    flat은 np.int32 형이며, 음수도 2's complement 그대로 하위 wbit 비트만 사용한다고 가정.
    """
    total_elems = flat.size
    total_bits = total_elems * wbit
    num_int32 = (total_bits + 31) // 32  # 올림

    packed = np.zeros(num_int32, dtype=np.int32)
    bit_offset = 0

    for i in range(total_elems):
        # 하위 wbit비트만 사용
        v = flat[i] & ((1 << wbit) - 1)
        bits_remaining = wbit

        while bits_remaining > 0:
            current_int_index = bit_offset // 32
            in_word_offset = bit_offset % 32
            space_in_current_word = 32 - in_word_offset

            bits_to_write = space_in_current_word
            if bits_to_write > bits_remaining:
                bits_to_write = bits_remaining

            data_chunk = v & ((1 << bits_to_write) - 1)
            packed[current_int_index] |= (data_chunk << in_word_offset)

            v >>= bits_to_write
            bits_remaining -= bits_to_write
            bit_offset += bits_to_write

    return packed

@njit
def _unpack_bits_jit(packed: np.ndarray, wbit: int, total_elems: int) -> np.ndarray:
    """
    wbit 단위로 패킹된 int32 배열 packed를 풀어, 길이=total_elems 인 1D np.int32로 복원.
    """
    unp = np.zeros(total_elems, dtype=np.int32)
    bit_offset = 0

    for i in range(total_elems):
        bits_remaining = wbit
        val = 0
        shift_amount = 0

        while bits_remaining > 0:
            current_int_index = bit_offset // 32
            in_word_offset = bit_offset % 32
            space_in_current_word = 32 - in_word_offset

            bits_to_read = space_in_current_word
            if bits_to_read > bits_remaining:
                bits_to_read = bits_remaining

            chunk = packed[current_int_index]
            chunk >>= in_word_offset
            chunk &= (1 << bits_to_read) - 1

            val |= chunk << shift_amount

            shift_amount += bits_to_read
            bits_remaining -= bits_to_read
            bit_offset += bits_to_read

        unp[i] = val

    return unp


# ----------------------
# [2] Pack / Unpack (PyTorch) 함수
# ----------------------

def pack_tensor(tensor: torch.Tensor, wbit: int):
    """
    torch.Tensor (int8/int16/int32) -> (packed_int32 : torch.Tensor, original_shape)
    로 패킹한다.

    * wbit in [1..32]
    * 특수 케이스 wbit=8,16,32 & 텐서 dtype이 대응 & 길이 조건이 맞으면
      -> 빠른 path(메모리 view) 사용 (3,4번 최적화)
    * 나머지는 fallback(numba) 방식
    """
    # 1) CPU 텐서로 복사 (Numba 등 처리 위해)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    original_shape = tuple(tensor.shape)
    num_elems = tensor.numel()

    # ========== [A] 특수 케이스 (wbit=8,16,32 + dtype 매칭 + 길이조건) ==========
    #    - wbit=8  => 4개 int8 -> 1개 int32
    #    - wbit=16 => 2개 int16 -> 1개 int32
    #    - wbit=32 => 1개 int32 -> 1개 int32 (사실상 no-op)
    # ==========================================================================
    if wbit == 32 and tensor.dtype == torch.int32:
        # 그냥 int32를 일렬화해서 반환 (실질적으로 "packing"이 필요 없음)
        flat_i32 = tensor.flatten()  # 이미 int32
        return flat_i32, original_shape

    if wbit == 16 and tensor.dtype == torch.int16:
        # 2개의 int16 => 1개의 int32
        # 길이 체크: num_elems % 2 == 0
        if num_elems % 2 == 0:
            flat_i16 = tensor.view(torch.int16)  # 이미 int16
            # NumPy view
            arr_i16 = flat_i16.numpy()  # shape=(num_elems,)
            # int16 -> int32 reinterpret (2바이트씩 4바이트로 합침)
            # 주의: 리틀 엔디안 기준으로 [i, i+1] 2개 int16이 순서대로 32비트 하나가 됨
            arr_i32 = arr_i16.view(np.int32)
            packed_torch = torch.from_numpy(arr_i32)  # shape=(num_elems//2,) int32
            return packed_torch, original_shape
        # else: 길이가 맞지 않으면 fallback

    if wbit == 8 and tensor.dtype == torch.int8:
        # 4개의 int8 => 1개의 int32
        # 길이 체크: num_elems % 4 == 0
        if num_elems % 4 == 0:
            flat_i8 = tensor.view(torch.int8)
            arr_i8 = flat_i8.numpy()  # shape=(num_elems,)
            arr_i32 = arr_i8.view(np.int32)
            packed_torch = torch.from_numpy(arr_i32)  # shape=(num_elems//4,) int32
            return packed_torch, original_shape
        # else: fallback

    # ========== [B] Fallback (Numba) ==========
    #   - 위의 특수 케이스가 아니거나, 길이가 맞지 않는 경우
    # ==========================================
    # (1) torch -> np.int32로 캐스팅
    flat_i32 = tensor.flatten().to(torch.int32).numpy()
    # (2) numba 함수 호출
    packed_np = _pack_bits_jit(flat_i32, wbit)
    packed_torch = torch.from_numpy(packed_np)  # int32 1D
    return packed_torch, original_shape


def unpack_tensor(packed: torch.Tensor, wbit: int, original_shape: tuple) -> torch.Tensor:
    """
    pack_tensor로 패킹된 결과(packed)와 wbit, 원본 shape를 받아,
    다시 torch.Tensor(dtype=int32) 혹은 (특수 케이스면) 원본 dtype으로 복원.
    """
    # 1) CPU로
    if packed.device.type != 'cpu':
        packed = packed.cpu()

    # 2) 원본 총 개수
    total_elems = 1
    for s in original_shape:
        total_elems *= s

    # ========== [A] 특수 케이스 복원 ==========
    #  original_dtype 판단:
    #   - (wbit=32, orig was int32) -> packed 자체가 flatten된 int32
    #   - (wbit=16, orig was int16) -> packed.shape=(total_elems//2,)
    #   - (wbit=8,  orig was int8 ) -> packed.shape=(total_elems//4,)
    # 여기서는 original_dtype이 무엇이었는지 따로 저장해야 하지만,
    # 예시는 "wbit=16->int16", "wbit=8->int8", "wbit=32->int32"라고 가정
    # 원본 dtype을 기록해두려면, pack_tensor 리턴값을 조금 수정하거나
    # 별도 프로토콜로 저장하면 됨.
    # 여기선 예시로 "wbit=8->torch.int8, wbit=16->torch.int16, wbit=32->torch.int32"라 가정.

    # 우선 original_dtype을 간단히 매핑
    def guess_dtype_from_wbit(wbit_):
        if wbit_ == 32:
            return torch.int32
        elif wbit_ == 16:
            return torch.int16
        elif wbit_ == 8:
            return torch.int8
        else:
            return torch.int32  # fallback일 경우 여기서 딱히 맞출 방법이 없음(패킹 시 정보 필요)

    orig_dtype = guess_dtype_from_wbit(wbit)

    if (wbit == 32 and orig_dtype == torch.int32):
        # 그냥 packed 자체가 원소수=total_elems 인 int32
        # shape=(total_elems,)
        if packed.numel() == total_elems:
            return packed.view(original_shape)  # 복원 완료

    if (wbit == 16 and orig_dtype == torch.int16):
        # 2개의 int16 -> 1개의 int32
        # packed.shape=(total_elems//2,)
        if packed.numel() * 2 == total_elems:
            arr_i32 = packed.numpy()  # shape=(total_elems//2,)
            arr_i16 = arr_i32.view(np.int16)  # shape=(total_elems,)
            out_tensor = torch.from_numpy(arr_i16).view(original_shape)
            return out_tensor

    if (wbit == 8 and orig_dtype == torch.int8):
        # 4개의 int8 -> 1개의 int32
        # packed.shape=(total_elems//4,)
        if packed.numel() * 4 == total_elems:
            arr_i32 = packed.numpy()  # shape=(total_elems//4,)
            arr_i8 = arr_i32.view(np.int8)   # shape=(total_elems,)
            out_tensor = torch.from_numpy(arr_i8).view(original_shape)
            return out_tensor

    # ========== [B] Fallback (Numba) ==========
    packed_i32 = packed.to(torch.int32).numpy()  # 1D
    unp_i32 = _unpack_bits_jit(packed_i32, wbit, total_elems)
    # 원본이 음수 포함했으면 2's complement 하위 wbit로부터 그대로 가져왔으므로
    # 실제 복원 dtype은 int32로 가정 (혹은 필요시 다른 타입으로 캐스팅)
    out_tensor = torch.from_numpy(unp_i32).view(original_shape)
    # 여기선 "원본이 int8/16"이었다 해도 몇 비트만 썼는지 정보가 없으니
    # 안전하게 int32로 반환. (만약 정말 int8/16으로 되돌리고 싶다면 추가 캐스팅)
    return out_tensor

# ---------------------- [테스트 / 사용 예시] ----------------------
if __name__ == "__main__":
    # 1) 큰 텐서 (4096,4096) / wbit=8 / dtype=int8 (최적화 케이스)
    shape = (4096, 4096)
    wbit_example = 8
    # -128 ~ 127 범위이지만 여기서는 단순하게 0~127 (무작위)로 예시
    original = torch.randint(low=0, high=128, size=shape, dtype=torch.int8)

    print("[Case 1] shape={}, wbit={}".format(original.shape, wbit_example))
    print("Packing...")
    packed, orig_shape = pack_tensor(original, wbit_example)
    print("  packed.shape =", packed.shape, "/ dtype =", packed.dtype)

    print("Unpacking...")
    recovered = unpack_tensor(packed, wbit_example, orig_shape)
    print("  recovered.shape =", recovered.shape, "/ dtype =", recovered.dtype)

    print("검증(원본 == 복원):", bool(torch.equal(original, recovered)))

    print()
    # 2) 큰 텐서 (2048,4096) / wbit=5 / dtype=int32 (일반 fallback)
    shape2 = (768, 768)
    wbit_example2 = 4
    original2 = torch.randint(low=-8, high=8, size=shape2, dtype=torch.int32)  # 예: -16 ~ +15 범위
    print("[Case 2] shape={}, wbit={}".format(original2.shape, wbit_example2))
    print("Packing...(fallback numba)")
    packed2, orig_shape2 = pack_tensor(original2, wbit_example2)
    print("  packed2.shape =", packed2.shape)

    print("Unpacking...(fallback numba)")
    recovered2 = unpack_tensor(packed2, wbit_example2, orig_shape2)
    print("  recovered2.shape =", recovered2.shape, "/ dtype =", recovered2.dtype)

    print("검증(원본 == 복원):", bool(torch.equal(original2, recovered2)))
