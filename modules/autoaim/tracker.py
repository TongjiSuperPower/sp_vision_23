from collections.abc import Iterable
from modules.autoaim.armor import Armor
from modules.autoaim.targets.outpost import Outpost


max_lost_count = 100
min_detect_count = 3


class Tracker:
    def __init__(self) -> None:
        self.target: Outpost = None
        self.state = 'LOST'

    def init(self, armors: Iterable[Armor], img_time_s: float) -> None:
        # 按近远排序，同时将armors从Iterable转换为list
        armors = sorted(armors, key=lambda a: a.in_camera_mm[2])

        if len(armors) == 0:
            return

        # 优先打最近的
        armor = armors[0]

        self.state = 'DETECTING'
        self._target_name = armor.name
        self._last_time_s = img_time_s
        self._lost_count = 0
        self._detect_count = 1

        if self._target_name == 'small_outpost':
            self.target = Outpost()

        self.target.init(armor)

    def update(self, armors: Iterable[Armor], img_time_s: float) -> None:
        dt_s = img_time_s - self._last_time_s
        self._last_time_s = img_time_s
        self.target.predict(dt_s)

        # 按近远排序，同时将armors从Iterable转换为list
        target_armors = filter(lambda a: a.name == self._target_name, armors)
        target_armors = sorted(target_armors, key=lambda a: a.in_camera_mm[2])

        matched = False
        if len(target_armors) > 0:
            matched = True
            self.target.update(target_armors[0])

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
