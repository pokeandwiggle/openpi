import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_rtc_delay_probs_accepts_a_distribution():
    config = _pi0_config.Pi0Config(pi05=True, rtc_delay_probs=[0.25, 0.0, 0.0, 0.0, 0.75])
    # Normalized to a tuple so the config stays hashable when built from parsed JSON.
    assert config.rtc_delay_probs == (0.25, 0.0, 0.0, 0.0, 0.75)


def test_rtc_delay_probs_rejects_bad_distributions():
    with pytest.raises(ValueError, match="pi05=True"):
        _pi0_config.Pi0Config(rtc_delay_probs=(0.5, 0.5))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _pi0_config.Pi0Config(pi05=True, rtc_delay=2, rtc_delay_probs=(0.5, 0.5))
    with pytest.raises(ValueError, match="entries"):
        _pi0_config.Pi0Config(pi05=True, rtc_delay_probs=(1.0,))
    with pytest.raises(ValueError, match="entries"):
        _pi0_config.Pi0Config(pi05=True, action_horizon=4, rtc_delay_probs=(0.0, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        _pi0_config.Pi0Config(pi05=True, rtc_delay_probs=(0.75, -0.25, 0.5))
    with pytest.raises(ValueError, match="last entry"):
        _pi0_config.Pi0Config(pi05=True, rtc_delay_probs=(0.5, 0.5, 0.0))
    with pytest.raises(ValueError, match="sum to 1"):
        _pi0_config.Pi0Config(pi05=True, rtc_delay_probs=(0.5, 0.4))


def _tiny_config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=8,
        max_token_len=16,
        **overrides,
    )


def test_sample_actions_without_a_prefix_matches_the_unconditioned_model():
    # A checkpoint trained with rtc_delay_probs saw the unconditioned (delay-0) mode on every
    # example whose sampled delay was 0. Serving that mode is a per-call choice — no prefix —
    # and must run exactly the baseline sampling path.
    key = jax.random.key(0)
    obs = _tiny_config().fake_obs(1)
    noise = jax.random.normal(jax.random.key(1), (1, 8, 4))

    rtc_model = _tiny_config(rtc_delay=2).create(key)
    baseline_model = _tiny_config().create(key)

    unprefixed = rtc_model.sample_actions(key, obs, num_steps=2, noise=noise)
    baseline = baseline_model.sample_actions(key, obs, num_steps=2, noise=noise)
    assert jnp.array_equal(unprefixed, baseline)


def test_sample_actions_returns_a_given_prefix_verbatim():
    key = jax.random.key(0)
    model = _tiny_config(rtc_delay=2).create(key)
    obs = _tiny_config().fake_obs(1)
    prefix = jnp.zeros((1, 8, 4)).at[:, :2].set(0.5)

    actions = model.sample_actions(key, obs, num_steps=2, action_prefix=prefix)

    assert jnp.array_equal(actions[:, :2], prefix[:, :2])


def test_sample_actions_refuses_a_prefix_without_rtc_delay():
    key = jax.random.key(0)
    model = _tiny_config().create(key)
    obs = _tiny_config().fake_obs(1)

    with pytest.raises(ValueError, match="rtc_delay"):
        model.sample_actions(key, obs, num_steps=1, action_prefix=jnp.zeros((1, 8, 4)))
