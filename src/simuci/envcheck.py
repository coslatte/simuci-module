"""Shim for backward compatibility: python -m simuci.envcheck"""

from simuci.tooling.envcheck import main

if __name__ == "__main__":
    main()
