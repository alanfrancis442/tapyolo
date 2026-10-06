from tapyolo.device import batch_for_device, resolve_device


def test_cuda_request_falls_back_when_cuda_is_absent():
    assert resolve_device("cuda:0", cuda=False, mps=False) == "cpu"
    assert resolve_device("cuda:0", cuda=False, mps=True) == "mps"
    assert resolve_device("cuda:0", cuda=True, mps=False) == "cuda:0"
    assert resolve_device("cpu", cuda=True, mps=True) == "cpu"
    assert resolve_device("mps", cuda=False, mps=False) == "cpu"


def test_batch_matches_the_vram_budget():
    assert batch_for_device("cuda:0") == 8
    assert batch_for_device("cpu") == 2
    assert batch_for_device("cuda:0", requested=4) == 4
