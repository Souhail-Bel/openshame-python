import mediapipe as mp
import cv2
import numpy as np
import math
import time
import random
import gc

# --- Constants ---
# Cooldowns & Thresholds
CLICK_COOLDOWN_FRAMES = 15
PINCH_THRESHOLD = 0.05
THUMB_TO_PALM_THRESHOLD = 0.07

# Physics Parameters
MAX_PHYSICS_OBJECTS = 75
NORMAL_FIST_RANGE = 250
NORMAL_PALM_RANGE = 200
WIDER_FIST_RANGE = 400
WIDER_PALM_RANGE = 350
MAX_PHYSICS_SPEED = 40
GRAVITY = np.array([0.0, 0.6])
MAX_LOGO_DIMENSION = 100 # Max width or height for spawned logos

# --- Global State & Buffers ---
# State
click_cooldown = 0
fist_menu_active, highlighted_dial_option = False, -1
frame_count = 0

# A centralized dictionary to hold all reusable image buffers for effects.
EFFECT_BUFFERS = {
    "ghosting_trail": None,
    "ascii": None,
    "neon_glow": None,
    "orb_glow": None,
    "blur": None # A generic buffer for blur operations
}

# UI & Physics States
face_buttons = [
    {"label": "Cartoon", "is_on": False, "center": (0, 0)},
    {"label": "ASCII", "is_on": False, "center": (0, 0)},
    {"label": "Ghosting", "is_on": False, "center": (0, 0)},
    {"label": "Thermal", "is_on": False, "center": (0, 0)},
]
dial_options = [
    {"label": "Normal Range", "is_on": True},
    {"label": "Wider Range", "is_on": False}
]
physics_objects = []
left_hand_grab, right_hand_grab = None, None
LOGO_IMAGES = [] # List to hold pre-loaded and resized logo images

# --- MediaPipe Initialization ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- Gesture Detection & State ---
def get_gesture_state(hand_landmarks):
    """Analyzes hand landmarks once and returns a dictionary of boolean gesture states."""
    if not hand_landmarks:
        return {
            "is_peace": False, "is_thumb_to_palm": False, "is_pinching": False,
            "is_open_palm": False, "is_fist": False
        }

    def is_finger_extended(tip_idx, pip_idx):
        return hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y

    def is_finger_curled(tip_idx, mcp_idx):
        return hand_landmarks.landmark[tip_idx].y > hand_landmarks.landmark[mcp_idx].y

    is_pinching_val = math.hypot(
        hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x - hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y - hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y
    ) < PINCH_THRESHOLD

    fingers_extended = all([
        is_finger_extended(mp_holistic.HandLandmark.INDEX_FINGER_TIP, mp_holistic.HandLandmark.INDEX_FINGER_PIP),
        is_finger_extended(mp_holistic.HandLandmark.MIDDLE_FINGER_TIP, mp_holistic.HandLandmark.MIDDLE_FINGER_PIP),
        is_finger_extended(mp_holistic.HandLandmark.RING_FINGER_TIP, mp_holistic.HandLandmark.RING_FINGER_PIP),
        is_finger_extended(mp_holistic.HandLandmark.PINKY_TIP, mp_holistic.HandLandmark.PINKY_MCP)
    ])

    fingers_curled = all([
        is_finger_curled(mp_holistic.HandLandmark.INDEX_FINGER_TIP, mp_holistic.HandLandmark.INDEX_FINGER_MCP),
        is_finger_curled(mp_holistic.HandLandmark.MIDDLE_FINGER_TIP, mp_holistic.HandLandmark.MIDDLE_FINGER_MCP),
        is_finger_curled(mp_holistic.HandLandmark.RING_FINGER_TIP, mp_holistic.HandLandmark.RING_FINGER_MCP),
        is_finger_curled(mp_holistic.HandLandmark.PINKY_TIP, mp_holistic.HandLandmark.PINKY_MCP)
    ])
    
    is_peace_val = (
        is_finger_extended(mp_holistic.HandLandmark.INDEX_FINGER_TIP, mp_holistic.HandLandmark.INDEX_FINGER_PIP) and
        is_finger_extended(mp_holistic.HandLandmark.MIDDLE_FINGER_TIP, mp_holistic.HandLandmark.MIDDLE_FINGER_PIP) and
        is_finger_curled(mp_holistic.HandLandmark.RING_FINGER_TIP, mp_holistic.HandLandmark.RING_FINGER_MCP) and
        is_finger_curled(mp_holistic.HandLandmark.PINKY_TIP, mp_holistic.HandLandmark.PINKY_MCP)
    )

    return {
        "is_peace": is_peace_val,
        "is_thumb_to_palm": math.hypot(
            hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x - hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x,
            hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y - hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y
        ) < THUMB_TO_PALM_THRESHOLD,
        "is_pinching": is_pinching_val,
        "is_open_palm": fingers_extended,
        "is_fist": fingers_curled
    }

# --- Buffer Management ---
def get_buffer(name, shape, dtype=np.uint8):
    """Gets a reusable buffer for stateless effects, creating or resizing it if necessary."""
    buffer = EFFECT_BUFFERS.get(name)
    if buffer is None or buffer.shape != shape or buffer.dtype != dtype:
        EFFECT_BUFFERS[name] = np.zeros(shape, dtype=dtype)
    else:
        EFFECT_BUFFERS[name][:] = 0
    return EFFECT_BUFFERS[name]

# --- Physics Sandbox ---
def spawn_physics_object(shape='circle', image_data=None):
    if len(physics_objects) >= MAX_PHYSICS_OBJECTS:
        physics_objects.pop(0)

    x = random.randint(100, 1180)
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    
    obj = {'pos': np.array([float(x), 50.0]), 
           'prev_pos': np.array([float(x), 50.0]),
           'vel': np.array([random.uniform(-2, 2), 0.0]), 
           'shape': shape, 
           'color': color, 
           'held_by': None,
           'image': None}

    if shape == 'image' and image_data is not None:
        obj['image'] = image_data
        obj['size'] = max(image_data.shape[0], image_data.shape[1]) // 2
    else:
        obj['size'] = random.randint(20, 40)
        
    physics_objects.append(obj)

def update_and_draw_physics(image, results, gestures):
    global left_hand_grab, right_hand_grab
    h, w, _ = image.shape

    fist_range = WIDER_FIST_RANGE if dial_options[1]['is_on'] else NORMAL_FIST_RANGE
    palm_range = WIDER_PALM_RANGE if dial_options[1]['is_on'] else NORMAL_PALM_RANGE

    for hand_name, hand_landmarks in [('left', results.left_hand_landmarks), ('right', results.right_hand_landmarks)]:
        if not hand_landmarks: continue
        palm_center = np.array([hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x * w, hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y * h])
        
        if gestures[hand_name]["is_fist"]:
            cv2.circle(image, tuple(palm_center.astype(int)), fist_range, (255, 150, 50), 2)
            for obj in physics_objects:
                dist_vec = palm_center - obj['pos']
                dist = np.linalg.norm(dist_vec)
                if 1 < dist < fist_range: obj['vel'] += dist_vec / dist * (1 - dist / fist_range) * 10
        elif gestures[hand_name]["is_open_palm"]:
            cv2.circle(image, tuple(palm_center.astype(int)), palm_range, (50, 150, 255), 2)
            for obj in physics_objects:
                dist_vec = obj['pos'] - palm_center
                dist = np.linalg.norm(dist_vec)
                if 1 < dist < palm_range: obj['vel'] += dist_vec / dist * (1 - dist / palm_range) * 18

    hand_grabs = {'left': left_hand_grab, 'right': right_hand_grab}
    for hand_name, hand_landmarks in [('left', results.left_hand_landmarks), ('right', results.right_hand_landmarks)]:
        if gestures[hand_name]["is_pinching"]:
            if not hand_grabs[hand_name]:
                thumb_tip, index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP], hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
                pinch_pos = np.array([(thumb_tip.x + index_tip.x) / 2 * w, (thumb_tip.y + index_tip.y) / 2 * h])
                for obj in sorted(physics_objects, key=lambda o: np.linalg.norm(o['pos'] - pinch_pos)):
                    if np.linalg.norm(obj['pos'] - pinch_pos) < obj['size'] and not obj['held_by']:
                        obj['held_by'], hand_grabs[hand_name] = hand_name, obj
                        break
        elif hand_grabs[hand_name]:
            hand_grabs[hand_name]['held_by'], hand_grabs[hand_name] = None, None
    left_hand_grab, right_hand_grab = hand_grabs['left'], hand_grabs['right']

    for obj in physics_objects:
        if obj['held_by']:
            hand_landmarks = results.left_hand_landmarks if obj['held_by'] == 'left' else results.right_hand_landmarks
            if hand_landmarks:
                thumb_tip, index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP], hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
                obj['pos'] = np.array([(thumb_tip.x + index_tip.x) / 2 * w, (thumb_tip.y + index_tip.y) / 2 * h])
                obj['vel'] = (obj['pos'] - obj['prev_pos']) * 0.9
            else:
                hand_that_disappeared = obj['held_by']
                obj['held_by'] = None
                if hand_that_disappeared == 'left': left_hand_grab = None
                else: right_hand_grab = None
        else:
            obj['vel'] = (obj['vel'] + GRAVITY) * 0.98
            speed = np.linalg.norm(obj['vel'])
            if speed > MAX_PHYSICS_SPEED: obj['vel'] = (obj['vel'] / speed) * MAX_PHYSICS_SPEED
            obj['pos'] += obj['vel']
            if not (obj['size'] <= obj['pos'][0] <= w - obj['size']): obj['vel'][0] *= -0.7
            if not (obj['size'] <= obj['pos'][1] <= h - obj['size']): obj['vel'][1] *= -0.7
            obj['pos'][0], obj['pos'][1] = np.clip(obj['pos'][0], obj['size'], w - obj['size']), np.clip(obj['pos'][1], obj['size'], h - obj['size'])

        obj['prev_pos'] = obj['pos'].copy()
        pos_int = obj['pos'].astype(np.int32)
        
        shape_drawers = {
            'circle': lambda: cv2.circle(image, tuple(pos_int), obj['size'], obj['color'], -1),
            'square': lambda: cv2.rectangle(image, (pos_int[0]-obj['size'], pos_int[1]-obj['size']), (pos_int[0]+obj['size'], pos_int[1]+obj['size']), obj['color'], -1),
        }
        if obj['shape'] in shape_drawers:
            shape_drawers[obj['shape']]()
        elif obj['shape'] == 'triangle':
            s = obj['size']
            points = np.array([[pos_int[0], pos_int[1] - s], [pos_int[0] - int(s * 0.866), pos_int[1] + int(s * 0.5)], [pos_int[0] + int(s * 0.866), pos_int[1] + int(s * 0.5)]], np.int32)
            cv2.fillPoly(image, [points], obj['color'])
        elif obj['shape'] == 'star':
            s_outer, s_inner = obj['size'], obj['size'] // 2
            points = np.array([[int(pos_int[0] + (s_outer if i%2==0 else s_inner) * math.cos(i*math.pi/5 - math.pi/2)), int(pos_int[1] + (s_outer if i%2==0 else s_inner) * math.sin(i*math.pi/5 - math.pi/2))] for i in range(10)], np.int32)
            cv2.fillPoly(image, [points], obj['color'])
        elif obj['shape'] == 'image' and obj.get('image') is not None:
            img_to_draw = obj['image']
            img_h, img_w, _ = img_to_draw.shape
            
            x1, y1 = pos_int[0] - img_w // 2, pos_int[1] - img_h // 2
            x2, y2 = x1 + img_w, y1 + img_h

            if x1 < w and y1 < h and x2 > 0 and y2 > 0:
                roi = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                alpha = img_to_draw[:, :, 3] / 255.0
                alpha = alpha[:roi.shape[0], :roi.shape[1]]
                
                for c in range(0, 3):
                    img_c = img_to_draw[:roi.shape[0], :roi.shape[1], c]
                    roi[:, :, c] = (img_c * alpha) + (roi[:, :, c] * (1.0 - alpha))


# --- Effects Functions ---
def apply_thermal_vision_effect(image):
    """Applies a thermal vision effect to the entire image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- MODIFIED: Changed the colormap to JET ---
    # You can experiment with other colormaps like:
    # cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_LAVA
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Blend the thermal image with the original to retain some detail
    superimposed_img = cv2.addWeighted(thermal, 0.6, image, 0.4, 0)
    
    return superimposed_img

def draw_geometric_orb_effect(image, hand_landmarks):
    if not hand_landmarks: return
    orb_glow_buffer = get_buffer("orb_glow", image.shape)
    blur_buffer = get_buffer("blur", image.shape)

    center_x = int(hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x * image.shape[1])
    center_y = int(hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y * image.shape[0])
    t = time.time()
    core_radius = int(15 * (1 + 0.5 * math.sin(t * 5)))
    core_color = (255, 255, int(200 * (1 + math.sin(t * 5)) / 2))
    cv2.circle(image, (center_x, center_y), core_radius, core_color, -1)
    
    polygons = [(3, 60, t * 2, (0, 255, 255)), (6, 90, -t, (255, 0, 255))]
    for sides, radius, angle, color in polygons:
        points = np.array([[int(center_x + radius * math.cos(angle + i * 2 * math.pi / sides)), int(center_y + radius * math.sin(angle + i * 2 * math.pi / sides))] for i in range(sides)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(orb_glow_buffer, [points], True, color, 20)
        cv2.polylines(image, [points], True, (255, 255, 255), 2)
    
    cv2.GaussianBlur(orb_glow_buffer, (31, 31), 0, dst=blur_buffer)
    image[:] = cv2.addWeighted(image, 1, blur_buffer, 0.8, 0)

def draw_neon_outline_glow(image, results):
    neon_glow_buffer = get_buffer("neon_glow", image.shape)
    blur_buffer = get_buffer("blur", image.shape)

    neon_color, line_thickness = (100, 255, 255), 4
    landmark_sets = [(results.pose_landmarks, mp_holistic.POSE_CONNECTIONS), (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS), (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)]
    for landmarks, connections in landmark_sets:
        if landmarks:
            mp_drawing.draw_landmarks(neon_glow_buffer, landmarks, connections, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=neon_color, thickness=line_thickness))
    
    cv2.GaussianBlur(neon_glow_buffer, (21, 21), 0, dst=blur_buffer)
    image[:] = cv2.addWeighted(image, 1, blur_buffer, 0.8, 0)
    image[:] = cv2.addWeighted(image, 1, neon_glow_buffer, 1, 0)
    return image

def apply_dream_sequence_effect(image, results):
    dimmed_image = cv2.convertScaleAbs(image, alpha=0.4, beta=0)
    neon_image = draw_neon_outline_glow(dimmed_image, results)
    if results.left_hand_landmarks: draw_geometric_orb_effect(neon_image, results.left_hand_landmarks)
    if results.right_hand_landmarks: draw_geometric_orb_effect(neon_image, results.right_hand_landmarks)
    h, w, _ = image.shape
    for cx, cy in [(50, 50), (w - 50, 50), (50, h - 50), (w - 50, h - 50)]:
        radius = int(30 * (1 + 0.2 * math.sin(time.time() * 3 + cx)))
        cv2.circle(neon_image, (cx, cy), radius, (255, 100, 255), 2)
    return neon_image

def apply_cartoon_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_ascii_art_effect(image):
    ascii_buffer = get_buffer("ascii", image.shape)
    ASCII_CHARS = " .:-=+*#%@"
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_h, small_w = 45, 80
    resized_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    cell_h, cell_w = h // small_h, w // small_w
    for y in range(small_h):
        for x in range(small_w):
            intensity = resized_gray[y, x]
            char_index = int(intensity / 255 * (len(ASCII_CHARS) - 1))
            cv2.putText(ascii_buffer, ASCII_CHARS[char_index], (x * cell_w, y * cell_h + cell_h), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
    return ascii_buffer

# --- UI Functions ---
def draw_and_handle_face_ui(image, results, gestures):
    global click_cooldown
    if not results.face_landmarks: return
    h, w, _ = image.shape
    face_buttons[0]["center"] = (int(results.face_landmarks.landmark[234].x * w) - 80, int(results.face_landmarks.landmark[234].y * h))
    face_buttons[1]["center"] = (int(results.face_landmarks.landmark[454].x * w) + 80, int(results.face_landmarks.landmark[454].y * h))
    face_buttons[2]["center"] = (int(results.face_landmarks.landmark[152].x * w), int(results.face_landmarks.landmark[152].y * h) + 60)
    face_buttons[3]["center"] = (int(results.face_landmarks.landmark[10].x * w), int(results.face_landmarks.landmark[10].y * h) - 60)

    if click_cooldown == 0:
        for hand_name in ['left', 'right']:
            if gestures[hand_name]["is_pinching"]:
                hand_landmarks = results.left_hand_landmarks if hand_name == 'left' else results.right_hand_landmarks
                ix = int(hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * w)
                iy = int(hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * h)
                for button in face_buttons:
                    if math.hypot(ix - button["center"][0], iy - button["center"][1]) < 30:
                        button["is_on"] = not button["is_on"]
                        click_cooldown = CLICK_COOLDOWN_FRAMES
                        break
                if click_cooldown > 0: break
    
    for button in face_buttons:
        center = button["center"]
        color = (0, 255, 255) if button["is_on"] else (150, 150, 150)
        cv2.circle(image, center, 30, color, 4)
        cv2.circle(image, center, 30, (255, 255, 255), 1)
        if button["is_on"]:
            cv2.circle(image, center, 18, (255, 255, 255), -1)
        text_size = cv2.getTextSize(button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(image, button["label"], (center[0] - text_size[0]//2, center[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_and_handle_fist_arc_menu(image, hand_landmarks, gestures):
    global fist_menu_active, highlighted_dial_option
    is_fist = gestures["right"]["is_fist"]
    is_open_palm = gestures["right"]["is_open_palm"]

    if is_fist: fist_menu_active = True
    elif is_open_palm and fist_menu_active:
        if highlighted_dial_option != -1:
            for i, option in enumerate(dial_options): option["is_on"] = (i == highlighted_dial_option)
        fist_menu_active = False
    elif not is_fist: fist_menu_active = False
    
    if not fist_menu_active:
        highlighted_dial_option = -1
        return

    h, w, _ = image.shape
    mcp_landmarks = [hand_landmarks.landmark[lm] for lm in [mp_holistic.HandLandmark.INDEX_FINGER_MCP, mp_holistic.HandLandmark.MIDDLE_FINGER_MCP, mp_holistic.HandLandmark.RING_FINGER_MCP]]
    anchor_x = int(np.mean([lm.x for lm in mcp_landmarks]) * w)
    anchor_y = int(np.mean([lm.y for lm in mcp_landmarks]) * h - 60)
    wrist, mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST], hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]
    roll_angle = math.atan2(mcp.y - wrist.y, mcp.x - wrist.x)
    
    arc_radius, num_options, arc_span_deg = 120, len(dial_options), 90
    start_angle_deg = -90 - arc_span_deg / 2
    cv2.ellipse(image, (anchor_x, anchor_y), (arc_radius, arc_radius), 0, start_angle_deg, start_angle_deg + arc_span_deg, (100, 100, 100), 3)

    neutral_angle, sensitivity_rad = -math.pi / 2, math.pi / 4
    angle_diff = roll_angle - neutral_angle
    selector_value = np.clip((angle_diff + sensitivity_rad) / (2 * sensitivity_rad), 0, 1)
    selector_angle_deg = start_angle_deg + selector_value * arc_span_deg
    dot_x, dot_y = int(anchor_x + arc_radius * math.cos(math.radians(selector_angle_deg))), int(anchor_y + arc_radius * math.sin(math.radians(selector_angle_deg)))

    option_data = []
    for i in range(num_options):
        angle_deg = start_angle_deg + (i / max(1, num_options - 1)) * arc_span_deg
        opt_x = int(anchor_x + arc_radius * math.cos(math.radians(angle_deg)))
        opt_y = int(anchor_y + arc_radius * math.sin(math.radians(angle_deg)))
        option_data.append({'pos': (opt_x, opt_y), 'angle_deg': angle_deg})

    distances = [math.hypot(dot_x - p['pos'][0], dot_y - p['pos'][1]) for p in option_data]
    highlighted_dial_option = np.argmin(distances)
    
    for i, option in enumerate(dial_options):
        opt_x, opt_y = option_data[i]['pos']
        color = (0, 255, 0) if option["is_on"] else (200, 200, 200)
        if i == highlighted_dial_option: cv2.circle(image, (opt_x, opt_y), 15, (255, 255, 0), -1)
        cv2.circle(image, (opt_x, opt_y), 10, color, -1)
        
        text_radius = arc_radius + 45
        angle_rad = math.radians(option_data[i]['angle_deg'])
        text_x = int(anchor_x + text_radius * math.cos(angle_rad))
        text_y = int(anchor_y + text_radius * math.sin(angle_rad))
        
        label = option["label"]
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_origin = (text_x - text_size[0] // 2, text_y + text_size[1] // 2)
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.circle(image, (dot_x, dot_y), 8, (0, 255, 255), -1)

def draw_all_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255,165,0), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(128,0,128), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(0,128,128), thickness=2, circle_radius=2))

# --- Main Application Loop ---
def main():
    global click_cooldown, frame_count, LOGO_IMAGES
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Interactive Gesture Control', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Interactive Gesture Control', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Loading logo images...")
    for i in range(1, 9):
        try:
            logo_path = f'logo_{i}.png'
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise FileNotFoundError(f"Image not found at {logo_path}")
            if logo.shape[2] != 4:
                logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)

            h, w = logo.shape[:2]
            scale = MAX_LOGO_DIMENSION /  max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            LOGO_IMAGES.append(resized_logo)
            print(f"-> Successfully loaded and resized logo_{i}.png")
        except Exception as e:
            print(f"Warning: Could not load 'logo_{i}.png'. {e}")
            LOGO_IMAGES.append(None)
    
    try:
        logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
        logo_resized = cv2.resize(logo, (200, int(200 * logo.shape[0] / logo.shape[1]))) if logo is not None else None
    except Exception as e:
        print(f"Warning: Could not load 'logo.png'. {e}")
        logo_resized = None

    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Camera frame is empty, skipping.")
                continue

            frame_count += 1
            if click_cooldown > 0: click_cooldown -= 1

            flipped_frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic_model.process(rgb_frame)
            display_image = flipped_frame.copy()

            gestures = {
                "left": get_gesture_state(results.left_hand_landmarks),
                "right": get_gesture_state(results.right_hand_landmarks)
            }

            special_effect_active = gestures["left"]["is_peace"] and gestures["right"]["is_peace"]
            if special_effect_active:
                display_image = apply_dream_sequence_effect(display_image, results)
            else:
                if face_buttons[3]["is_on"]:
                    display_image = apply_thermal_vision_effect(display_image)
                else:
                    if face_buttons[0]["is_on"]: display_image = apply_cartoon_effect(display_image)
                    if face_buttons[1]["is_on"]: display_image = apply_ascii_art_effect(display_image)
                    
                    if face_buttons[2]["is_on"]:
                        ghosting_trail = EFFECT_BUFFERS.get("ghosting_trail")
                        if ghosting_trail is None or ghosting_trail.shape != display_image.shape:
                            EFFECT_BUFFERS["ghosting_trail"] = display_image.astype(np.float32)
                        else:
                            current_frame_float = display_image.astype(np.float32)
                            EFFECT_BUFFERS["ghosting_trail"] = cv2.addWeighted(current_frame_float, 0.1, ghosting_trail, 0.9, 0)
                        display_image = EFFECT_BUFFERS["ghosting_trail"].astype(np.uint8)
                    else:
                        EFFECT_BUFFERS["ghosting_trail"] = None

            update_and_draw_physics(display_image, results, gestures)
            
            if not special_effect_active:
                if gestures["left"]["is_thumb_to_palm"]: draw_geometric_orb_effect(display_image, results.left_hand_landmarks)
                if gestures["right"]["is_thumb_to_palm"]: draw_geometric_orb_effect(display_image, results.right_hand_landmarks)
                if results.right_hand_landmarks: draw_and_handle_fist_arc_menu(display_image, results.right_hand_landmarks, gestures)
                if not face_buttons[3]["is_on"]:
                    draw_all_landmarks(display_image, results)
            
            draw_and_handle_face_ui(display_image, results, gestures)
            
            if logo_resized is not None:
                h, w, _ = display_image.shape
                if logo_resized.shape[2] == 4:
                    overlay_h, overlay_w, _ = logo_resized.shape
                    roi = display_image[20:20+overlay_h, w-overlay_w-20:w-20]
                    alpha = logo_resized[:,:,3] / 255.0
                    for c in range(0,3):
                        roi[:,:,c] = logo_resized[:,:,c] * alpha + roi[:,:,c] * (1.0 - alpha)

            text_to_display = "PRESS: b/s/t/x/1-8: spawn | c: clear | q: quit"
            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            outline_thickness, text_color, outline_color = 4, (255, 255, 255), (0, 0, 0)
            text_pos = (20, 40)
            cv2.putText(display_image, text_to_display, text_pos, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
            cv2.putText(display_image, text_to_display, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.imshow('Interactive Gesture Control', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('b'): spawn_physics_object('circle')
            elif key == ord('s'): spawn_physics_object('square')
            elif key == ord('t'): spawn_physics_object('triangle')
            elif key == ord('x'): spawn_physics_object('star')
            elif key == ord('c'): physics_objects.clear()
            elif ord('1') <= key <= ord('8'):
                logo_index = key - ord('1')
                if logo_index < len(LOGO_IMAGES) and LOGO_IMAGES[logo_index] is not None:
                    spawn_physics_object(shape='image', image_data=LOGO_IMAGES[logo_index])
            
            if frame_count % 1000 == 0:
                gc.collect()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down...")
        holistic_model.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()