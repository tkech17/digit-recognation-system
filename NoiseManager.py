from random import random
from random import seed

from utils.AudioFileMultiplier import addNoisedAudios

if __name__ == "__main__":
    seed(1)
    addNoisedAudios(0.01, str(random() * 10000))
