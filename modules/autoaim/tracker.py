import numpy as np
from collections.abc import Iterable
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target
from modules.autoaim.targets.outpost import Outpost


max_match_distance_m = 0.1
max_lost_count = 100
min_detect_count = 3


class Tracker:
    def __init__(self) -> None:
        self.target: Target = None
        self.state = 'LOST'

    def init(self, armors: Iterable[Armor], img_time_s: float) -> None:
        # 按近远排序，同时将armors从Iterable转换为list
        armors = sorted(armors, key=lambda a: a.in_camera_mm[2])

        if len(armors) == 0:
            return

        # 优先打最近的
        armor = armors[0]

        self.state = 'DETECTING'
        self._lost_count = 0
        self._detect_count = 1

        if armor.name == 'small_outpost':
            self.target = Outpost(armor, img_time_s)

    def update(self, armors: Iterable[Armor], img_time_s: float) -> None:
        self.target.predict_to(img_time_s)
        predicted_armor_position_m = self.target.get_armor_position_m()

        min_distance_m = np.inf
        min_distance_armor: Armor = None
        for armor in filter(lambda a: a.name == self.target.name, armors):
            distance_m = np.linalg.norm(predicted_armor_position_m - armor.in_imu_m)
            if distance_m < min_distance_m:
                min_distance_m = distance_m
                min_distance_armor = armor

        matched = False

        if min_distance_m < max_match_distance_m:
            matched = True
            self.target.update(min_distance_armor)

        elif min_distance_armor is not None:
            matched = True
            self.target.handle_armor_jump(min_distance_armor)


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
