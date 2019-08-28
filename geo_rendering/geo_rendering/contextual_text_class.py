import cv2


class ContextualText:

    def __init__(self, content, position, color):
        self.text_content = content
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_style = cv2.LINE_AA
        self.position = position
        self.color = color
        self.text_size = 1
        self.thickness = 1

    def display_text(self, map_to_edit):
        """
            Renders the text object on the provided image file, 
            with all its attributes as parameters
        """
        text = self.text_content
        pos = self.position
        col = self.color
        font = self.font
        size = self.text_size
        thick = self.thickness
        style = self.font_style

        cv2.putText(map_to_edit, text, pos, font, size, col, thick, style)