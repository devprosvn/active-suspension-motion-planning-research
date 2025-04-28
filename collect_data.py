#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirSim v1.2.0 – Data Collector for the “MountainLandscape” map
• Python ≥ 3.7        • pygame 2.1.3
• Supports forward & reverse driving
• Captures RGB frames from three cameras at a fixed rate

Keys
────
W / ↑   : drive forward
S / ↓   : reverse (automatic gear--1)
A / ←   : steer left
D / →   : steer right
R       : start / pause recording
Q       : quit
"""

# ────────────────────────── imports ──────────────────────────
import os, time, cv2, numpy as np, pandas as pd
from datetime import datetime
import airsim
import pygame

# ───────────────────────── settings check ────────────────────
REQ_PYGAME_VER = (2, 1, 3)
if pygame.version.vernum[:3] != REQ_PYGAME_VER:
    print(f"⚠️  Recommended pygame {'.'.join(map(str, REQ_PYGAME_VER))}, "
          f"but running {pygame.version.ver}.  "
          f"If keyboard input misbehaves, install with "
          f"`pip install pygame==2.1.3`.")

# ─────────────────────── DataCollector ───────────────────────
class DataCollector:
    def __init__(self, save_dir: str = "collected_data"):
        # 1 – connect to AirSim
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.reset()

        # 2 – directories
        self.save_dir   = save_dir
        self.images_dir = os.path.join(save_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # 3 – buffers
        self.data_log = []                 # list of dicts

        # 4 – cameras
        self.cam_names = ["center", "left", "right"]
        self.cam_ids   = {"center": 0, "left": 1, "right": 2}

        print("✅ DataCollector ready (Python 3.7+, pygame 2.1.3).")

    # ───────────────────── frame capture ──────────────────────
    def collect_frame(self):
        """Grab one frame from the three cameras and append to the log."""
        state   = self.client.getCarState()
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        throttle = self.cur_ctrls.throttle
        steering = self.cur_ctrls.steering
        speed    = state.speed

        # request three images in one call
        reqs  = [airsim.ImageRequest(self.cam_ids[c], airsim.ImageType.Scene,
                                     False, False) for c in self.cam_names]
        resps = self.client.simGetImages(reqs)

        imgs = {}
        for name, resp in zip(self.cam_names, resps):
            buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            if buf.size == 0:
                print(f"⚠️ Empty frame from {name}; skipping.")
                return None
            rgb = buf.reshape(192, 256, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fname = f"{name}_{stamp}.png"
            cv2.imwrite(os.path.join(self.images_dir, fname), bgr)
            imgs[name] = fname

        self.data_log.append(dict(timestamp=stamp,
                                  center_image=imgs["center"],
                                  left_image  =imgs["left"],
                                  right_image =imgs["right"],
                                  steering=steering,
                                  throttle=throttle,
                                  speed=speed))
        return imgs

    # ─────────────────────── helpers ──────────────────────────
    @staticmethod
    def _init_pygame():
        pygame.init()
        pygame.display.set_caption("AirSim Keyboard Control – pygame 2.1.3")
        screen = pygame.display.set_mode((420, 90))   # small HUD
        font   = pygame.font.SysFont("consolas", 16)
        return screen, font

    def _save_csv(self):
        if not self.data_log:
            print("⚠️ No data captured.")
            return
        out = os.path.join(self.save_dir, "driving_log.csv")
        pd.DataFrame(self.data_log).to_csv(out, index=False)
        print(f"💾 Log saved → {out}")

    # ───────────────── manual collection ──────────────────────
    def manual_data_collection(self, duration_s: int = 300, freq_hz: int = 10):
        screen, font = self._init_pygame()

        # initialise vehicle controls
        self.cur_ctrls = airsim.CarControls()
        self.cur_ctrls.handbrake = False
        self.client.setCarControls(self.cur_ctrls)

        key_on = dict(fwd=False, back=False, left=False, right=False)
        recording = False
        print("W/A/S/D or arrows to drive | R to record | Q to quit")

        dt      = 1.0 / freq_hz
        t_next  = time.time() + dt
        t_end   = time.time() + duration_s

        try:
            while time.time() < t_end:

                # ---------- keyboard events ----------
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if ev.type in (pygame.KEYDOWN, pygame.KEYUP):
                        pressed = ev.type == pygame.KEYDOWN
                        if ev.key in (pygame.K_w, pygame.K_UP):
                            key_on["fwd"] = pressed
                        elif ev.key in (pygame.K_s, pygame.K_DOWN):
                            key_on["back"] = pressed
                        elif ev.key in (pygame.K_a, pygame.K_LEFT):
                            key_on["left"] = pressed
                        elif ev.key in (pygame.K_d, pygame.K_RIGHT):
                            key_on["right"] = pressed
                        elif ev.key == pygame.K_r and pressed:
                            recording = not recording
                        elif ev.key == pygame.K_q and pressed:
                            raise KeyboardInterrupt
                # --------------------------------------

                # ---------- vehicle control logic ----------
                c = self.cur_ctrls   # alias for brevity

                # steering (works both forward & reverse)
                c.steering = (-0.5 if key_on["left"]
                              else 0.5 if key_on["right"]
                              else 0.0)

                if key_on["back"]:                 # ---- reverse mode ----
                    c.is_manual_gear = True
                    c.manual_gear    = -1          # gear -1 = reverse
                    c.throttle       = 0.5
                    c.brake          = 0.0
                elif key_on["fwd"]:                # ---- drive forward ---
                    c.is_manual_gear = False       # auto gear
                    c.throttle       = 0.7
                    c.brake          = 0.0
                else:                              # ---- idle / stop -----
                    c.is_manual_gear = False
                    c.throttle       = 0.0
                    c.brake          = 0.2         # light brake to hold

                self.client.setCarControls(c)
                # ----------------------------------------

                # ---------- logging ----------
                if recording and time.time() >= t_next:
                    self.collect_frame()
                    t_next += dt

                # ---------- HUD ----------
                gear = "R" if c.is_manual_gear and c.manual_gear == -1 else "D"
                status = "REC" if recording else "PAUSE"
                hud = f"{status} | {gear}  thr={c.throttle:.1f}  brk={c.brake:.1f}  steer={c.steering:.1f}"
                screen.fill((40, 40, 40))
                screen.blit(font.render(hud,
                                        True,
                                        (0, 220, 0) if recording else (220, 220, 0)),
                            (10, 30))
                pygame.display.flip()
                pygame.time.wait(10)   # ≈ 100 Hz main loop

        except KeyboardInterrupt:
            print("⏹️ Stopped by user.")

        finally:
            # ensure safe stop & save log
            self.cur_ctrls.throttle = 0
            self.cur_ctrls.brake    = 1
            self.cur_ctrls.steering = 0
            self.client.setCarControls(self.cur_ctrls)
            self._save_csv()
            pygame.quit()

    # ───────────────────────── cleanup ─────────────────────────
    def cleanup(self):
        self.client.enableApiControl(False)
        print("✅ API control released.")

# ──────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    dc = DataCollector()
    try:
        dc.manual_data_collection(duration_s=300, freq_hz=10)
    finally:
        dc.cleanup()