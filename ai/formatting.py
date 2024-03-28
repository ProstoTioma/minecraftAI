import pyautogui
import time

# Define the duration to hold each key (adjust as needed)
key_hold_duration = 0.1


class Game:
    def __init__(self, predicted_movement):
        # Simulate pressing the keys based on predicted movements
        for movement in predicted_movement:
            if movement == "w":
                pyautogui.keyDown("w")
            elif movement == "a":
                pyautogui.keyDown("a")
            elif movement == "s":
                pyautogui.keyDown("s")
            elif movement == "d":
                pyautogui.keyDown("d")

            # Wait for a short duration before releasing the key (optional)
            time.sleep(key_hold_duration)

            # Release the key
            pyautogui.keyUp(movement)
