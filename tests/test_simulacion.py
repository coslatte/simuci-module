"""Tests for simuci.simulacion — Simulation SimPy process."""

from __future__ import annotations

import simpy

from simuci import Experiment, Simulation
from simuci.internals._constants import EXPERIMENT_VARIABLES_LABELS


class TestSimulation:
    """Verify that the SimPy process runs and populates results."""

    def _run_simulation(self, experiment: Experiment, cluster: int) -> dict[str, int]:
        env = simpy.Environment()
        experiment.init_results_variables()
        sim = Simulation(experiment, cluster)
        env.process(sim.uci(env))
        env.run()
        return experiment.result

    def test_cluster_0_produces_results(self, experiment: Experiment) -> None:
        result = self._run_simulation(experiment, cluster=0)
        assert set(result.keys()) == set(EXPERIMENT_VARIABLES_LABELS.keys())
        assert all(isinstance(v, int) for v in result.values())

    def test_cluster_1_produces_results(self, experiment: Experiment) -> None:
        result = self._run_simulation(experiment, cluster=1)
        assert set(result.keys()) == set(EXPERIMENT_VARIABLES_LABELS.keys())

    def test_vam_does_not_exceed_uci(self, experiment: Experiment) -> None:
        for cluster in (0, 1):
            for _ in range(20):
                r = self._run_simulation(experiment, cluster)
                assert r["vam"] <= r["uci"]

    def test_time_invariant(self, experiment: Experiment) -> None:
        """pre_vam + vam + post_vam == uci."""

        for cluster in (0, 1):
            for _ in range(10):
                r = self._run_simulation(experiment, cluster)
                assert r["pre_vam"] + r["vam"] + r["post_vam"] == r["uci"]

    def test_simulation_stores_on_experiment(self, experiment: Experiment) -> None:
        """The Simulation object writes to experiment.result, not its own state."""

        self._run_simulation(experiment, cluster=0)
        assert len(experiment.result) == 5
