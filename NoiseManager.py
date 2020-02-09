from random import Random

from utils.AudioFileMultiplier import addNoisedAudios

if __name__ == "__main__":
    addNoisedAudios(0.01, str(Random.randint(1, 10000000)))
