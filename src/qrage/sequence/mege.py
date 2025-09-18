from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
from pypulseq import Opts, Sequence

from .readout import ReadoutKernel
from .sampling import PartitionSampling


class MEGE:
    """
    TBD
    """

    def __init__(
        self,
        fov: Tuple[int, int, int],
        matrix_size: Tuple[int, int, int],
        axes: SimpleNamespace,
        readout_bandwidth: float = 390.625,
        num_spokes: int = 8,
        num_echoes: int = 9,
        num_partitions_per_block: int = 16,
        num_autocalibration_lines: int = 32,
        acceleration_factor: int = 2,
        debug: bool = False,
        system: Union[Opts, None] = None,
    ) -> None:
        """
        Initialize the QRAGE class.

        Parameters:
        ----------
        fov : Tuple[int, int, int]
            Field of view in millimeters (x, y, z).
        matrix_size : Tuple[int, int, int]
            Matrix size (x, y, z).
        axes : SimpleNamespace
            Axis mapping for the encoding directions.
        readout_bandwidth : float
            The readout bandwidth in kHz.
        num_spokes : int
            The number of spokes per acquisition.
        num_sets : int
            The number of sets in the acquisition.
        num_echoes : int
            The number of echoes per acquisition.
        num_partitions_per_block : int
            The number of partitions to sample per block.
        num_autocalibration_lines : int
            The number of autocalibration lines.
        acceleration_factor : int
            The acceleration factor for partition sampling.
        debug : bool
            If True, enables debug mode.
        system : Opts, default=Opts()
            System limits.
        """
        num_partitions_per_block = matrix_size[axes.n3]
        num_sets = 1

        self.sampling = PartitionSampling(
            matrix_size[axes.n3],
            num_partitions_per_block,
            num_autocalibration_lines,
            acceleration_factor,
        )
        self.readout_kernel = ReadoutKernel(
            fov,
            matrix_size,
            axes,
            readout_bandwidth=readout_bandwidth,
            num_sets=num_sets,
            num_echoes=num_echoes,
            debug=debug,
            system=system,
        )

        self.num_spokes = num_spokes
        self.num_sets = num_sets
        self.num_echoes = num_echoes
        self.num_partitions_per_block = num_partitions_per_block

        # (TODO) Analytic timing calculation is currently not correct

        # self.dTI = (
        #     self.readout_kernel.total_time * self.sampling.num_partitions_per_block
        # )
        # self.TR = self.inversion_kernel.total_time + self.dTI * self.num_sets

        self.TI0_delay = 0
        self.dTI_delay = 0
        self.TR_delay = 0

    def run(
        self,
        seq: Sequence,
    ) -> None:
        """
        Runs the QRAGE sequence by executing the inversion and readout kernels
        for each partition.

        Parameters:
        ----------
        seq : Sequence
            Sequence object.
        """
        self.readout_kernel.prep(seq)
        self.readout_kernel.reset(seq)

        index_block = 0
        index_set = 0

        for index_phase in range(self.num_spokes):
            for index_partition_in_block in range(
                self.sampling.num_partitions_per_block
            ):
                index_partition = self.sampling.partition(
                    index_block, index_partition_in_block
                )
                self.readout_kernel.run(
                    seq, index_phase, int(index_partition), index_set
                )

    def get_timing(
        self,
        seq: Sequence,
    ) -> None:
        """
        Get timing information of the QRAGE sequence.

        Parameters:
        ----------
        seq : Sequence
            Sequence object.
        """

        t_excitation, _, _, _, _, _ = seq.rf_times()
        t_adc, _ = seq.adc_times()

        t_excitation = np.asarray(t_excitation) * 1e3
        t_adc = np.asarray(t_adc) * 1e3

        t_adc = t_adc.reshape(
            (-1, self.num_sets, self.num_partitions_per_block, self.num_echoes, 256)
        )
        t_excitation = t_excitation.reshape(
            (-1, self.num_sets, self.num_partitions_per_block)
        )

        t_adc_mean = np.mean(t_adc, axis=-1)

        self.TE = t_adc_mean[0, 0, 0, :] - t_excitation[0, 0, 0]

        self.TE0 = self.TE[0]
        self.dTE = self.TE[1] - self.TE[0]
