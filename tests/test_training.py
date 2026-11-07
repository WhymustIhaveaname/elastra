import json
from collections import Counter

import torch

from elastra import config, residual
from elastra.training import (
    _cut_history,
    environment_surfaces,
    stage_position_observation,
    stage_surfaces,
)


def test_host_curriculum_replaces_the_stiff_mattresses_from_update_500():
    cfg = config.load(["train=host"], config_name="training")
    surfaces = environment_surfaces(cfg.train, int(cfg.train.seed))
    assert len(surfaces) == cfg.train.num_envs
    before = Counter(stage_surfaces(cfg.train, surfaces, 499))
    after = Counter(stage_surfaces(cfg.train, surfaces, 500))
    assert before == Counter({k: int(v) for k, v in cfg.train.surfaces.items()})
    for old, new in (("a16", "a1"), ("a32", "a2"), ("a64", "a4")):
        assert after[f"mattress/{new}"] == before[f"mattress/{old}"]
        assert after[f"mattress/{old}"] == 0
    changed = [
        (a, b)
        for a, b in zip(
            stage_surfaces(cfg.train, surfaces, 0), stage_surfaces(cfg.train, surfaces, 500)
        )
        if a != b
    ]
    assert len(changed) == 96


def test_position_observation_starts_with_the_second_host_stage():
    host = config.load(["train=host"], config_name="training")
    base = host.residuals.host.position_observation
    assert not stage_position_observation(host.train, base, 499)
    assert stage_position_observation(host.train, base, 500)
    pm = config.load(["train=protomotions"], config_name="training")
    assert pm.train.updates == 500 and not pm.train.curriculum
    pm_base = pm.residuals.protomotions.position_observation
    assert not stage_position_observation(pm.train, pm_base, 0)


def test_grown_networks_reproduce_the_narrow_ones():
    actor, critic = residual.build_actor_critic(85, 23, seed=283402)
    obs = torch.randn(48, 92)
    narrow = torch.cat([obs[:, :84], obs[:, 91:]], dim=1)
    with torch.no_grad():
        mean, _ = actor(narrow)
        value = critic(narrow)
    residual.grow_networks(actor, critic, 84, 7)
    with torch.no_grad():
        grown_mean, _ = actor(obs)
        grown_value = critic(obs)
    torch.testing.assert_close(mean, grown_mean, rtol=0, atol=1e-6)
    torch.testing.assert_close(value, grown_value, rtol=0, atol=1e-5)


def test_grown_optimizer_state_starts_the_new_columns_at_zero():
    actor, critic = residual.build_actor_critic(85, 23, seed=1)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    loss = actor(torch.randn(8, 85))[0].square().sum() + critic(torch.randn(8, 85)).sum()
    loss.backward()
    optimizer.step()
    state = optimizer.state_dict()
    residual.grow_networks(actor, critic, 84, 7)
    grown = residual.grow_optimizer_state(state, actor, critic, 84, 7)
    params = list(actor.parameters()) + list(critic.parameters())
    first_layers = (actor.body[0].weight, critic.body[0].weight)
    firsts = [i for i, p in enumerate(params) if any(p is q for q in first_layers)]
    assert len(firsts) == 2
    for index in firsts:
        old = state["state"][index]["exp_avg"]
        new = grown["state"][index]["exp_avg"]
        assert new.shape == (old.shape[0], 92)
        assert torch.equal(new[:, 84:91], torch.zeros(old.shape[0], 7))
        assert torch.equal(torch.cat([new[:, :84], new[:, 91:]], dim=1), old)
        assert torch.equal(grown["state"][index]["step"], state["state"][index]["step"])


def test_cut_history_keeps_the_updates_up_to_the_checkpoint(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps({"update": u}) + "\n" for u in range(1, 8)))
    _cut_history(path, 5)
    assert [json.loads(line)["update"] for line in path.read_text().splitlines()] == [1, 2, 3, 4, 5]
