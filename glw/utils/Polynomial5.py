

class Polynomial5:
    def __init__(self) -> None:
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0

    def update_traj(self, pam:tuple, new_pos, set_time):
        new_acce = 0
        new_velo = 0

        now_pos = pam[0]
        now_velo = pam[1]
        now_acce = pam[2]

        h = new_pos - now_pos
        
        self.T = set_time
        # T_int = set_time * 1000.0
        # T = T_int / 1000.0
        T = self.T
        self.a0 = now_pos
        self.a1 = now_velo
        self.a2 = now_acce / 2.0
        self.a3 = (20.0 * h - (8 * new_velo + 12 * now_velo) * T - (3 * now_acce - new_acce) * T * T) / (2 * T * T * T)
        self.a4 = (-30.0 * h + (14 * new_velo + 16 * now_velo) * T + (3 * now_acce - 2 * new_acce) * T * T) / (2 * T * T * T * T)
        self.a5 = (12.0 * h - 6 * (new_velo + now_velo) * T + (new_acce - now_acce) * T * T) / (2 * T * T * T * T * T)

    def interpolation(self, t) -> tuple:
        if t < 0: t=0
        elif t > self.T: t = self.T

        p = self.a0 + self.a1 * (t) + self.a2 * (t * t) + self.a3 * (t * t * t) + self.a4 * (t * t * t * t) + self.a5 * (t * t * t * t * t);
        v = self.a1 + self.a2 * 2 * (t) + self.a3 * 3 * (t * t) + self.a4 * 4 * (t * t * t) + self.a5 * 5 * (t * t * t * t)
        a = self.a2 * 2 + self.a3 * 6 * (t) + self.a4 * 12 * (t * t) + self.a5 * 20 * (t * t * t)
        return (p, v, a)