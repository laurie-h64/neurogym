from typing import Any

import numpy as np

from neurogym.core import TrialEnv
from neurogym.utils import spaces
from neurogym.utils.ngym_random import TruncExp

class ContextDecisionMaking(TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities) and needs to make a perceptual decision based on one while
    ignoring the other. The task can operate in two modes:
    1. Implicit context: The relevant modality is fixed and must be learned (
       it is not explicitly signaled).
    2. Explicit context: The agent simultaneously receives stimulus inputs
       from two modalities, and the relevant modality is explicitly indicated
       by a context signal.
    Both modes use ring representation for encoding stimulus inputs and choices.

    Args:
        dt: Timestep of the environment in milliseconds.
        use_expl_context: If True, the context is explicit (signaled) and changes per trial.
            If False, a fixed context is used throughout (set by `impl_context_modality`).
        impl_context_modality: The fixed implicit modality to use if `use_expl_context`
            is False (0 or 1).
        dim_ring: Number of stimulus locations (or choices).
        rewards: Optional dictionary to override default rewards. The required keys are "abort" and "correct".
            Defaults to {"abort": -0.1, "correct": +1.0}.
        timing: Optional dictionary to override default durations of task periods.
            The expected keys are "fixation", "stimulus" (required), "delay", and "decision" (required).
            Defaults to {"fixation": 300, "stimulus": 750, "delay": TruncExp(600, 300, 3000), "decision": 100}.
        sigma: Standard deviation of Gaussian noise added to stimulus.
        abort: If True, incorrect actions during fixation abort the trial.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nature12742",
        "paper_name": """Context-dependent computation by recurrent
         dynamics in prefrontal cortex""",
        "tags": ["perceptual", "context dependent", "two-alternative", "supervised"],
    }

    def __init__(
        self,
        configuration: dict[str, str] | None = None,
        relevant_modality: int = 0,
        dt: int = 100,
        use_expl_context: bool = False,
        dim_ring: int = 2,
        rewards: dict[str, float] | None = None,
        timing: dict[str, int | TruncExp] | None = None,
        sigma: float = 1.0,
        abort: bool = False,
    ) -> None:
        super().__init__(dt=dt)


        # Configuration
        self.configuration = dict[str, str] = {
            'context_mode': 'random',
            'mod1_mode': 'graded',
            'mod2_mode': 'graded',
            }
        if configuration:
            self.configuration.update(configuration)

        self.context_mode = self.configuration['context_mode']
        self.mod1_mode = self.configuration['mod1_mode']
        self.mod2_mode = self.configuration['mod2_mode']

        # Specified relevant modality
        if relevant_modality not in [0, 1]:
            msg = "relevant_modality must be 0 or 1."
            raise ValueError(msg)

        self.relevant_modality = relevant_modality

        # Task parameters
        self.dt = dt
        self.use_expl_context = use_expl_context
        self.dim_ring = dim_ring
        self.sigma = sigma / np.sqrt(self.dt)

        # Trial conditions and spaces setup
        self.cohs_graded: list[int] = [5, 15, 50]
        self.cohs_categorical: list[int] = [0,100]
        self.cohs_absent: list[int] = [0]
        if use_expl_context:
            self.contexts: list[int] = [0, 1]
            if self.context_mode == 'fixed':
              self.contexts = [self.relevant_modality]
        else:
            self.contexts = [self.relevant_modality]
        self._setup_spaces()

        # Rewards
        self.rewards: dict[str, float] = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        self.timing: dict[str, int | TruncExp] = {
            "fixation": 300,
            "stimulus": 750,
            "delay": TruncExp(600, 300, 3000),
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.abort = abort

    def _setup_spaces(self):
        """Setup observation and action spaces using ring representation."""
        self.theta = np.linspace(0, 2 * np.pi, self.dim_ring + 1)[:-1]
        self.choices = np.arange(1, self.dim_ring + 1)

        # Base observation space with fixation and modalities
        obs_space_name = {
            "fixation": 0,
            **{f"stim{i + 1}_mod1": i + 1 for i in range(self.dim_ring)},
            **{f"stim{i + 1}_mod2": i + 1 + self.dim_ring for i in range(self.dim_ring)},
        }

        if self.use_expl_context:
            context_idx = 1 + 2 * self.dim_ring
            obs_space_name.update(
                {
                    "context1": context_idx,
                    "context2": context_idx + 1,
                },
            )

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(len(obs_space_name),),
            dtype=np.float32,
            name=obs_space_name,
        )

        # Action space: fixation + choices
        action_space_name = {"fixation": 0, "choice": list(range(1, self.dim_ring + 1))}

        self.action_space = spaces.Discrete(
            n=1 + self.dim_ring,
            name=action_space_name,
        )

    def _new_trial(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """New trial within the current context.

        Returns:
            dict: Trial information
        """
        # Trial parameters
        ground_truth = self.rng.choice(self.choices)
        other_choice = self.rng.choice(self.choices)

        context = self.rng.choice(self.contexts)

        if (self.mod1_mode == 'categorical') or (self.mod1_mode == 'graded_relevant' and context == 1):
            coh_1 = self.rng.choice(self.cohs_categorical)
        elif self.mod1_mode == 'absent':
            coh_1 = self.rng.choice(self.cohs_absent)
        else:
            coh_1 = self.rng.choice(self.cohs_graded)

        if (self.mod2_mode == 'categorical') or (self.mod2_mode == 'graded_relevant' and context == 0):
            coh_2 = self.rng.choice(self.cohs_categorical)
        elif self.mod2_mode == 'absent':
            coh_2 = self.rng.choice(self.cohs_absent)
        else:
            coh_2 = self.rng.choice(self.cohs_graded)

        trial = {
            "ground_truth": ground_truth,
            "other_choice": other_choice,
            "context": context,
            "coh_1": coh_1,
            "coh_2": coh_2,
            }
        trial.update(kwargs)

        choice_1, choice_2 = ground_truth, other_choice

        if context == 1: #mod2 is rel
          choice_1, choice_2 = choice_2, choice_1

        # Add periods
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)

        # Set observations based on context type
        self.add_ob(1, period="fixation", where="fixation")

        self._set_observations(choice_1, choice_2, coh_1, coh_2)
        # Add context signals for explicit context
        if self.use_expl_context:
            if trial["context"] == 0:
                self.add_ob(1, where="context1")
            else:
                self.add_ob(1, where="context2")
        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def _set_observations(self, choice_1, choice_2, coh_1, coh_2):
        """Set observations for ring representation."""
        stim_theta_1 = self.theta[choice_1 - 1]
        stim_theta_2 = self.theta[choice_2 - 1]

        stim_mod1 = np.cos(self.theta - stim_theta_1) * (coh_1 / 200) + 0.5
        stim_mod2 = np.cos(self.theta - stim_theta_2) * (coh_2 / 200) + 0.5

        # Add observations for each index
        for i in range(self.dim_ring):
            self.add_ob(stim_mod1[i], "stimulus", where=f"stim{i + 1}_mod1")
            self.add_ob(stim_mod2[i], "stimulus", where=f"stim{i + 1}_mod2")
            self.add_randn(
                0,
                self.sigma,
                period="stimulus",
                where=f"stim{i + 1}_mod1",
            )
            self.add_randn(
                0,
                self.sigma,
                period="stimulus",
                where=f"stim{i + 1}_mod2",
            )

        self.set_ob(0, "decision")

    def _step(self, action: int) -> tuple:  # type: ignore[override]
        """Execute one timestep within the environment."""
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        terminated = False
        truncated = False
        reward = 0.0

        if self.in_period("fixation") and action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
