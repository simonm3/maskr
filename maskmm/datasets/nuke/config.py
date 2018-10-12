from maskmm.config import Config

class Config(Config):
    NAME = "dsb"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    CLASS_NAMES = ["BG", "Cell"]
    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 30
