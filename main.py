from anomalias.detectors import Detectors
import anomalias.api as api


def main():
    detectors = Detectors()
    api.init(detectors)


if __name__ == "__main__":
    main()