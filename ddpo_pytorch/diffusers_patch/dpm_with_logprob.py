# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the UNet prediction.
# - Instead of `variance_noise`, it takes `prev_sample` as an optional argument. If `prev_sample` is provided,
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.

from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def dpm_step_with_logprob(
    self: DPMSolverMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
) -> Union[SchedulerOutput, Tuple]:

    """
    Step function propagating the sample with the multistep DPM-Solver.

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

    Returns:
        [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
        True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero()
    if len(step_index) == 0:
        step_index = len(self.timesteps) - 1
    else:
        step_index = step_index.item()
    prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
    lower_order_final = (
        (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
    )
    lower_order_second = (
        (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
    )

    model_output = self.convert_model_output(model_output, timestep, sample)
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output

    if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
        prev_sample = self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
    elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
        timestep_list = [self.timesteps[step_index - 1], timestep]
        prev_sample = self.multistep_dpm_solver_second_order_update(
            self.model_outputs, timestep_list, prev_timestep, sample
        )
    else:
        timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
        prev_sample = self.multistep_dpm_solver_third_order_update(
            self.model_outputs, timestep_list, prev_timestep, sample
        )

    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1

    if not return_dict:
        return (prev_sample,)

    # return SchedulerOutput(prev_sample=prev_sample), 0
    return prev_sample.type(sample.dtype), torch.zeros(prev_sample.shape[0])
