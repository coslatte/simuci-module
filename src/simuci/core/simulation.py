"""Discrete-event simulation process for a single ICU patient."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import simpy

from simuci.core import distributions

if TYPE_CHECKING:
    from simuci.core.experiment import Experiment


class Simulation:
    """SimPy process that models an ICU patient's journey through VAM stages."""

    def __init__(self, experiment: Experiment, cluster: int) -> None:
        self.experiment = experiment
        self.cluster = cluster

    def uci(self, env: simpy.Environment) -> Generator[simpy.Timeout, Any, None]:
        """Run the patient through pre-VAM → VAM → post-VAM → post-ICU stages."""

        # Select cluster-specific distribution functions
        if self.cluster == 0:
            post_uci_dist = distributions.tiemp_postUCI_cluster0
            uci_dist = distributions.estad_UTI_cluster0
            vam_dist = distributions.tiemp_VAM_cluster0
        else:
            post_uci_dist = distributions.tiemp_postUCI_cluster1
            uci_dist = distributions.estad_UTI_cluster1
            vam_dist = distributions.tiemp_VAM_cluster1

        # Sample from distributions
        post_uci = int(post_uci_dist())
        uci = int(uci_dist())

        # Ensure VAM ≤ UCI stay; retry up to 1000 draws then clamp
        for _ in range(1000):
            vam = int(vam_dist())
            if vam <= uci:
                break
        else:
            vam = uci

        pre_vam = int((uci - vam) * self.experiment.porcentaje / 100)
        post_vam = uci - pre_vam - vam

        self.experiment.result["pre_vam"] = pre_vam
        self.experiment.result["vam"] = vam
        self.experiment.result["post_vam"] = post_vam
        self.experiment.result["uci"] = uci
        self.experiment.result["post_uci"] = post_uci

        yield env.timeout(pre_vam)
        yield env.timeout(vam)
        yield env.timeout(post_vam)
        yield env.timeout(post_uci)
