class DetectedCard:
    def __init__(self, img_bgr, rect):
        self.img_bgr = img_bgr
        self.rect = rect

    @property
    def color(self):
        color = self.img_bgr.mean(axis=(0, 1))
        return color
    
    