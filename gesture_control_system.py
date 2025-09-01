import cv2, time, math, subprocess, sys, platform
import numpy as np
import pyautogui
import mediapipe as mp

OS = platform.system()
pyautogui.FAILSAFE = False

# ===== Hand tracking setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ===== Utility helpers =====

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Simple 1‑Euro style smoother
class Smoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev = None
    def __call__(self, x, y):
        if self.prev is None:
            self.prev = (x, y)
            return x, y
        sx = self.prev[0] + self.alpha*(x - self.prev[0])
        sy = self.prev[1] + self.alpha*(y - self.prev[1])
        self.prev = (sx, sy)
        return sx, sy

smooth = Smoother(alpha=0.25)

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# Gesture state
pinch_down = False
pinch_time = 0
last_swipe_ts = 0
swipe_cooldown = 0.6

# Click debounce
clicked = False

# ===== Platform actions =====

def os_mission_control():
    if OS == 'Darwin':
        # macOS Mission Control (Control+Up)
        pyautogui.hotkey('ctrl', 'up')
    elif OS == 'Windows':
        pyautogui.hotkey('win', 'tab')

def os_next_app():
    if OS == 'Darwin':
        pyautogui.keyDown('cmd'); pyautogui.press('tab'); pyautogui.keyUp('cmd')
    elif OS == 'Windows':
        pyautogui.keyDown('alt'); pyautogui.press('tab'); pyautogui.keyUp('alt')

def os_prev_app():
    if OS == 'Darwin':
        pyautogui.keyDown('cmd'); pyautogui.keyDown('shift'); pyautogui.press('tab'); pyautogui.keyUp('shift'); pyautogui.keyUp('cmd')
    elif OS == 'Windows':
        pyautogui.keyDown('alt'); pyautogui.keyDown('shift'); pyautogui.press('tab'); pyautogui.keyUp('shift'); pyautogui.keyUp('alt')

# macOS window snaps via AppleScript

def mac_snap(side):
    if OS != 'Darwin':
        return
    script_left = """
    tell application "System Events"
      tell application process (path to frontmost application as text)
        tell window 1
          set position to {0, 0}
          set size to { (item 1 of (get size of desktop)) / 2, item 2 of (get size of desktop) }
        end tell
      end tell
    end tell
    """
    script_right = """
    tell application "System Events"
      tell application process (path to frontmost application as text)
        tell window 1
          set {dw, dh} to size of desktop
          set position to {dw/2, 0}
          set size to {dw/2, dh}
        end tell
      end tell
    end tell
    """
    subprocess.run(["osascript", "-e", script_left if side=='left' else script_right])


def mac_minimize():
    if OS == 'Darwin':
        subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "m" using {command down}'])
    elif OS == 'Windows':
        pyautogui.hotkey('win', 'down')

# ===== Main loop =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Gesture control active — ESC to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        lm = hl.landmark
        # Landmarks of interest
        idx_tip = (int(lm[8].x*w), int(lm[8].y*h))
        thumb_tip = (int(lm[4].x*w), int(lm[4].y*h))
        mid_tip = (int(lm[12].x*w), int(lm[12].y*h))
        ring_tip = (int(lm[16].x*w), int(lm[16].y*h))
        pink_tip = (int(lm[20].x*w), int(lm[20].y*h))
        wrist = (int(lm[0].x*w), int(lm[0].y*h))

        # Draw hand
        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # ===== Air cursor =====
        # Map index tip to screen
        sx = int((idx_tip[0] / w) * SCREEN_W)
        sy = int((idx_tip[1] / h) * SCREEN_H)
        sx, sy = smooth(sx, sy)
        pyautogui.moveTo(sx, sy, duration=0)

        # ===== Gestures =====
        pinch_distance = dist(idx_tip, thumb_tip)
        mid_to_wrist = dist(mid_tip, wrist)
        ring_to_wrist = dist(ring_tip, wrist)
        pink_to_wrist = dist(pink_tip, wrist)

        # Heuristics
        pinch = pinch_distance < 40  # pixels in camera space; tweak as needed

        # Open palm: all fingers extended further from wrist than a threshold
        open_palm = min(mid_to_wrist, ring_to_wrist, pink_to_wrist) > 120

        # Fist: all fingertips close to wrist
        fist = max(mid_to_wrist, ring_to_wrist, pink_to_wrist) < 80

        # Two‑finger swipe detection (index+middle horizontal velocity)
        # Track simple velocity using previous index tip
        if not hasattr(pyautogui, '_prev_idx'):
            pyautogui._prev_idx = idx_tip
            pyautogui._prev_t = time.time()
        else:
            dt = time.time() - pyautogui._prev_t
            vx = (idx_tip[0] - pyautogui._prev_idx[0]) / max(dt, 1e-3)
            pyautogui._prev_idx, pyautogui._prev_t = idx_tip, time.time()
            # Consider middle finger near index = two‑finger swipe posture
            two_finger = dist(idx_tip, mid_tip) < 60
            if two_finger and abs(vx) > 900 and (time.time() - last_swipe_ts) > swipe_cooldown:
                if vx > 0:
                    os_next_app()
                else:
                    os_prev_app()
                last_swipe_ts = time.time()

        # Pinch → click / drag
        if pinch and not pinch_down:
            pinch_down = True
            pinch_time = time.time()
            pyautogui.mouseDown()
        elif pinch and pinch_down:
            # continue dragging
            pass
        elif not pinch and pinch_down:
            # release
            pyautogui.mouseUp()
            held = time.time() - pinch_time
            # quick tap counts as click
            if held < 0.20:
                pyautogui.click()
            pinch_down = False

        # Open palm → Mission Control / Task View (debounced)
        if open_palm and (time.time() - last_swipe_ts) > 0.8:
            os_mission_control()
            last_swipe_ts = time.time()

        # Fist → minimize
        if fist and (time.time() - last_swipe_ts) > 0.8:
            mac_minimize()
            last_swipe_ts = time.time()

        # L‑gesture snap: index horizontal away from thumb, with roughly 90° angle
        angle = math.degrees(math.atan2(idx_tip[1]-thumb_tip[1], idx_tip[0]-thumb_tip[0]))
        if open_palm is False and  -40 <= angle <= 20 and (time.time() - last_swipe_ts) > 0.8:
            mac_snap('right')
            last_swipe_ts = time.time()
        if open_palm is False and 160 <= angle or angle <= -160 and (time.time() - last_swipe_ts) > 0.8:
            mac_snap('left')
            last_swipe_ts = time.time()

    cv2.putText(frame, 'ESC to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()