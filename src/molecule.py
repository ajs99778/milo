from AaronTools.geometry import Geometry


class Molecule(Geometry):
    def __init__(
        self,
        *args,
        refresh_connected=False,
        refresh_ranks=False,
        spin_state=1,
        **kwargs
    ):
        super().__init__(
            *args,
            refresh_connected=refresh_connected,
            refresh_ranks=refresh_ranks,
            **kwargs
        )
        self.spin_state = spin_state
    