from typing_extensions import Literal

class WAE_MMD_Config():

    kernel_choice: Literal["rbf", "imq"] = "imq"
    reg_weight: float = 3e-2
    kernel_bandwidth: float = 1.0
    latent_dim: int = 16