"""Entry point for `python -m heulistic`.

The `heulistic` console script is the normal way in; this makes the module form
work too, which is how the CLI gets run inside a virtualenv that has not been
activated.
"""

from .main import main

if __name__ == "__main__":
    main()
