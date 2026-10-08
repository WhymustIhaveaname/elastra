import json
from collections import Counter

from elastra import config
from elastra.training import _cut_history, environment_surfaces, stage_surfaces


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


def test_protomotions_curriculum_matches_host():
    host = config.load(["train=host"], config_name="training").train
    pm = config.load(["train=protomotions_curriculum"], config_name="training").train
    assert pm.updates == host.updates == 1000
    assert pm.curriculum == host.curriculum
    assert pm.surfaces == host.surfaces


def test_cut_history_keeps_the_updates_up_to_the_checkpoint(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps({"update": u}) + "\n" for u in range(1, 8)))
    _cut_history(path, 5)
    assert [json.loads(line)["update"] for line in path.read_text().splitlines()] == [1, 2, 3, 4, 5]
