# keyboard_layout.py
# A comprehensive module for keyboard layout analysis, 3D path generation, and hex code conversion


# --- Base Keyboard Layout Class ---
class KeyboardLayout:
    """Base class for keyboard layouts with 3D coordinate mapping."""

    def __init__(self, layout_name):
        """Initialize a keyboard layout with a name."""
        self.layout_name = layout_name
        self.key_positions = {}  # {character: (x, y, z)}

    def set_key_position(self, char, x, y, z=0):
        """Assign 3D coordinates to a key on the keyboard layout."""
        self.key_positions[char.lower()] = (x, y, z)

    def get_key_position(self, char):
        """Get the 3D coordinates for a key."""
        return self.key_positions.get(char.lower())

    def get_3d_path(self, word):
        """Return a list of 3D coordinates for each character in the word."""
        path = []
        for char in word.lower():
            if char in self.key_positions:
                path.append(self.key_positions[char])
        return path

# --- QWERTY Layout ---
class QWERTYLayout(KeyboardLayout):
    """QWERTY keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("QWERTY")

        # Top row (y=0)
        self.set_key_position("`", x=0, y=0, z=0)
        for i, char in enumerate("1234567890-="):
            self.set_key_position(char, x=i+1, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("qwertyuiop[]\\"):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("asdfghjkl;'"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate("zxcvbnm,./"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("~", x=0, y=0, z=1)
        for i, char in enumerate("!@#$%^&*()_+"):
            self.set_key_position(char, x=i+1, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- Dvorak Layout ---
class DvorakLayout(KeyboardLayout):
    """Dvorak keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("Dvorak")

        # Top row (y=0)
        self.set_key_position("`", x=0, y=0, z=0)
        for i, char in enumerate("1234567890[]"):
            self.set_key_position(char, x=i+1, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("',.pyfgcrl/=\\"):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("aoeuidhtns-"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate(";qjkxbmwvz"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("~", x=0, y=0, z=1)
        for i, char in enumerate("!@#$%^&*(){}"):
            self.set_key_position(char, x=i+1, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- Colemak Layout ---
class ColemakLayout(KeyboardLayout):
    """Colemak keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("Colemak")

        # Top row (y=0)
        self.set_key_position("`", x=0, y=0, z=0)
        for i, char in enumerate("1234567890-="):
            self.set_key_position(char, x=i+1, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("qwfpgjluy;[]\\"):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("arstdhneio'"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate("zxcvbkm,./"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("~", x=0, y=0, z=1)
        for i, char in enumerate("!@#$%^&*()_+"):
            self.set_key_position(char, x=i+1, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- AZERTY Layout ---
class AZERTYLayout(KeyboardLayout):
    """AZERTY keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("AZERTY")

        # Top row (y=0)
        self.set_key_position("²", x=0, y=0, z=0)
        for i, char in enumerate("&é\"'(-è_çà)="):
            self.set_key_position(char, x=i+1, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("azertyuiop^$"):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("qsdfghjklmù*"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate("wxcvbn,;:!"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        for i, char in enumerate("1234567890°+"):
            self.set_key_position(char, x=i+1, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- QWERTZ Layout ---
class QWERTZLayout(KeyboardLayout):
    """QWERTZ keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("QWERTZ")

        # Top row (y=0)
        self.set_key_position("^", x=0, y=0, z=0)
        for i, char in enumerate("1234567890ß´"):
            self.set_key_position(char, x=i+1, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("qwertzuiopü+"):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("asdfghjklöä#"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate("yxcvbnm,.-"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("°", x=0, y=0, z=1)
        for i, char in enumerate("!\"§$%&/()=?`"):
            self.set_key_position(char, x=i+1, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- JIS Layout ---
class JISLayout(KeyboardLayout):
    """Japanese JIS keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("JIS")

        # Top row (y=0)
        for i, char in enumerate("1234567890-^¥"):
            self.set_key_position(char, x=i, y=0, z=0)

        # Middle row (y=1)
        for i, char in enumerate("qwertyuiop@["):
            self.set_key_position(char, x=i, y=1, z=0)

        # Lower row (y=2)
        for i, char in enumerate("asdfghjkl;:]"):
            self.set_key_position(char, x=i, y=2, z=0)

        # Bottom row (y=3)
        for i, char in enumerate("zxcvbnm,./\\"):
            self.set_key_position(char, x=i, y=3, z=0)

        # Special characters (shifted layer, z=1)
        for i, char in enumerate("!\"#$%&'()*+~|"):
            self.set_key_position(char, x=i, y=0, z=1)

        # Space bar
        self.set_key_position(" ", x=3, y=4, z=0)

# --- Right-Handed and Left-Handed Variants ---
class RightHandedLayout(KeyboardLayout):
    """Right-handed variant of a keyboard layout."""

    def __init__(self, base_layout, shift_amount=2):
        super().__init__(f"RightHanded{base_layout.layout_name}")
        self.base_layout = base_layout
        self.shift_amount = shift_amount

        # Shift all keys to the right
        for char, (x, y, z) in base_layout.key_positions.items():
            self.set_key_position(char, x=x+shift_amount, y=y, z=z)

class LeftHandedLayout(KeyboardLayout):
    """Left-handed variant of a keyboard layout."""

    def __init__(self, base_layout, shift_amount=2):
        super().__init__(f"LeftHanded{base_layout.layout_name}")
        self.base_layout = base_layout
        self.shift_amount = shift_amount

        # Shift all keys to the left
        for char, (x, y, z) in base_layout.key_positions.items():
            self.set_key_position(char, x=max(0, x-shift_amount), y=y, z=z)

# --- Path Generation and Hex Code Conversion ---
def generate_word_path(word, layout):
    """Return the 3D path for a word on a specific keyboard layout."""
    return layout.get_3d_path(word)

def project_orthogonal_3d_path(path):
    """Project a 3D path onto 6 orthogonal directions (2D/3D projections)."""
    front = [(x, y) for x, y, z in path]  # x-y plane
    back = [(x, y, 1) for x, y, z in path]  # x-y with z=1
    left = [(y, z) for x, y, z in path]  # y-z plane
    right = [(y, z, 10) for x, y, z in path]  # y-z with x=10
    top = [(x, z) for x, y, z in path]  # x-z plane
    bottom = [(x, z, 10) for x, y, z in path]  # x-z with y=10
    return {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }

def to_hex(projection):
    """Convert 2D/3D coordinates to hex code (4 or 6 hex digits per point)."""
    hex_code = []
    for point in projection:
        hex_digits = ""
        for axis in point:
            # Convert integer to 2 hex digits (00-FF)
            hex_axis = hex(axis)[2:].upper().zfill(2)
            hex_digits += hex_axis
        hex_code.append(hex_digits)
    return " ".join(hex_code)

# --- Example Usage ---
def example_word_analysis(word, layout):
    """Analyze a word's path on a keyboard layout and generate hex codes."""
    print(f"Analyzing word '{word}' on {layout.layout_name} layout:")

    # Generate 3D path
    path = generate_word_path(word, layout)
    print(f"3D Path: {path}")

    # Project to 6 orthogonal directions
    projections = project_orthogonal_3d_path(path)

    # Generate hex codes for all 6 views
    for view, projection in projections.items():
        print(f"{view.capitalize()} View Hex Code: {to_hex(projection)}")

    print()

# --- Main Function ---
def main():
    # Create layouts
    qwerty = QWERTYLayout()
    dvorak = DvorakLayout()
    colemak = ColemakLayout()
    azerty = AZERTYLayout()
    qwertz = QWERTZLayout()
    jis = JISLayout()

    # Create right-handed and left-handed variants
    right_hand_qwerty = RightHandedLayout(qwerty)
    left_hand_qwerty = LeftHandedLayout(qwerty)
    right_hand_jis = RightHandedLayout(jis)
    left_hand_jis = LeftHandedLayout(jis)

    # Example word
    word = "6orthogonal directions"

    # Analyze word on different layouts
    example_word_analysis(word, qwerty)
    example_word_analysis(word, dvorak)
    example_word_analysis(word, colemak)
    example_word_analysis(word, azerty)
    example_word_analysis(word, qwertz)
    example_word_analysis(word, jis)

    # Analyze word on right-handed and left-handed variants
    example_word_analysis(word, right_hand_qwerty)
    example_word_analysis(word, left_hand_qwerty)
    example_word_analysis(word, right_hand_jis)
    example_word_analysis(word, left_hand_jis)

if __name__ == "__main__":
    main()
