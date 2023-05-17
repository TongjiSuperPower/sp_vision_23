from numpy.typing import NDArray
from modules.autoaim.armor import Armor


class Target:
    name: str = None

    def __init__(self) -> None:
        raise NotImplementedError('该函数需子类实现')

    def init(self, armor: Armor, time_s: float) -> None:
        raise NotImplementedError('该函数需子类实现')

    def predict_to(self, time_s: float) -> None:
        raise NotImplementedError('该函数需子类实现')

    def get_armor_in_imu_m(self) -> NDArray:
        raise NotImplementedError('该函数需子类实现')

    def update(self, armor: Armor) -> None:
        raise NotImplementedError('该函数需子类实现')

    def handle_armor_jump(self, armor: Armor) -> None:
        raise NotImplementedError('该函数需子类实现')
