from anomalias.detectors import Detectors
import anomalias.api as api
import threading


def main():
    threading.current_thread().name = "anomalias"
    detectors = Detectors()
    api.init(detectors)


if __name__ == "__main__":
    main()