from collections.abc import Iterable
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target
from modules.autoaim.targets.standard import Standard
from modules.autoaim.targets.outpost import Outpost


max_lost_count = 50
min_detect_count = 5


class Tracker:
    def __init__(self) -> None:
        self.target: Target = None
        self.state = 'LOST'

    def init(self, armors: list[Armor], img_time_s: float) -> None:
        # 按近远排序，同时将armors从Iterable转换为list
        armors = sorted(armors, key=lambda a: a.in_camera_mm[2, 0])

        if len(armors) == 0:
            return

        # 优先打最近的
        armor = armors[0]

        if armor.name == 'small_outpost':
            self.target = Outpost()
        else:
            self.target = Standard()

        self.target.init(armor, img_time_s)

        self.state = 'DETECTING'
        self._target_name = armor.name
        self._lost_count = 0
        self._detect_count = 1

    def update(self, armors: list[Armor], img_time_s: float) -> None:
        self.target.predict(img_time_s)

        # 筛选装甲板
        target_armors = filter(lambda a: a.name == self._target_name, armors)
        # 按左右排序，同时将armors从Iterable转换为list
        target_armors = sorted(target_armors, key=lambda a: a.in_camera_mm[0, 0])

        matched = False
        reinit = False
        if len(target_armors) > 0:
            matched = True
            reinit = self.target.update(target_armors[0])

        # Tracker状态机
        if self.state == 'DETECTING':
            if matched:
                self._detect_count += 1
                if self._detect_count >= min_detect_count:
                    self._detect_count = 0
                    self.state = 'TRACKING'
            else:
                self._detect_count = 0
                self.state = 'LOST'

        elif self.state == 'TRACKING':
            if not matched:
                self.state = 'TEMP_LOST'
                self._lost_count += 1

        elif self.state == 'TEMP_LOST':
            if not matched:
                self._lost_count += 1
                if self._lost_count > max_lost_count:
                    self._lost_count = 0
                    self.state = 'LOST'
            else:
                self.state = 'TRACKING'
                self._lost_count = 0

        if reinit:
            self.state = 'DETECTING'
            self._lost_count = 0
            self._detect_count = 1
