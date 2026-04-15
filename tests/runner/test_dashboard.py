from rl_world_ascii.runner.dashboard import Dashboard, NullDashboard


def test_null_dashboard_methods_noop():
    d = NullDashboard()
    d.start()
    d.update_env("x", episodes_done=1, episodes_total=5, mean_return=0.5, mean_len=20.0)
    d.update_totals(cost_used=0.0, budget=None, parse_failures=0, provider_errors=0, fallback_noops=0)
    d.log_event("hello")
    d.stop()
    # No exceptions = pass


def test_dashboard_accumulates_stats():
    d = Dashboard(run_id="t", model_id="m", provider="mock", total_episodes=5)
    d.start()
    d.update_env("env1", episodes_done=1, episodes_total=3, mean_return=1.0, mean_len=10.0)
    d.update_env("env2", episodes_done=2, episodes_total=2, mean_return=2.0, mean_len=20.0)
    snapshot = d.snapshot()
    assert snapshot["env_rows"]["env1"]["episodes_done"] == 1
    assert snapshot["env_rows"]["env2"]["mean_return"] == 2.0
    d.stop()
