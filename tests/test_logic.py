from core import Veector
from virtual_space import VirtualSpace

def test_comparison():
    v = Veector()
    vs = VirtualSpace(v)
    tensor = [
        [[0], [0, 0, 0], [5, 3], 2],
        [[0], [0, 0, 0], [2, 0, 0], 1],  # >
        [1, 0, 0],
        [0, 1, 0],
        []
    ]
    vs.add_tensor(tensor)
    result = vs.execute(tensor)
    assert result == [1], f"Expected [1], got {result}"