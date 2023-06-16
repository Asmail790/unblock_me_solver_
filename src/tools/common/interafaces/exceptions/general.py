class GuiderException(Exception):
    """
    General exception indicating that can't guide.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
