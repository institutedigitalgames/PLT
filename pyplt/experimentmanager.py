# Preference Learning Toolbox
# Copyright (C) 2018 Institute of Digital Games, University of Malta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""This module contains the :class:`ExperimentManager` class which enables the batching of experiments."""

from pyplt.experiment import Experiment


class ExperimentManager:
    """Class for running a set of experiments in batch.

    The user may add any number of experiments to the experiment list. The experiments may then be run in batch.
    """

    _experiment_list = []

    def add_experiment(self, experiment: Experiment):  # experiment of type Experiment
        """Add an experiment to the list of experiments to be run in batch.

        :param experiment: the experiment to be added to the list.
        :type experiment: :class:`pyplt.experiment.Experiment`
        """
        self._experiment_list.append(experiment)

    def run_all(self):
        """Run each of the experiments in the list sequentially.

        :raises NoFeaturesError: if there are no features/attributes in the objects data of a given experiment.
        :raises NoRanksDerivedError: if rank derivation fails because no pairwise preferences could be derived
            from the given data of a given experiment. This is either
            because there are no clear pairwise preferences in the data or because none of the clear pairwise
            preferences in the data conform to the chosen values for the rank derivation parameters (i.e., the minimum
            distance margin (`mdm`) and the memory (`memory`) parameters).
        :raises InvalidParameterValueException: if the user attempted to use a negative value (i.e., smaller than 0.0)
            for the `mdm` rank derivation parameter of a given experiment.
        :raises NormalizationValueError: if normalization fails for a given experiment because one of the given values
            cannot be converted to int or float prior to the normalization.
        """
        # TODO: add threading to run multiple experiments at the same time (thus speeding up execution time)!!!
        for e in range(len(self._experiment_list)):
            self._experiment_list[e].run()
