from collections.abc import Iterable
from modules.ekf import ColumnVector
from modules.autoaim.armor import Armor


class Target:
    def __init__(self, armor: Armor, time_s: float) -> None:
        self.name: str = None
        raise NotImplementedError('该函数需子类实现')

    def predict_to(self, time_s: float) -> None:
        raise NotImplementedError('该函数需子类实现')

    def get_armor_position_m(self) -> ColumnVector:
        raise NotImplementedError('该函数需子类实现')

    def get_all_armor_positions_m(self) -> Iterable[ColumnVector]:
        raise NotImplementedError('该函数需子类实现')

    def update(self, armor: Armor) -> None:
        raise NotImplementedError('该函数需子类实现')

    def handle_armor_jump(self, armor: Armor) -> None:
        raise NotImplementedError('该函数需子类实现')
