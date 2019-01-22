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

"""This module contains the definitions for several exceptions, errors and warnings occurring specifically in PLT."""


class PLTException(Exception):
    """Base class for exceptions occurring in PLT.

    Extends the class :class:`Exception`.
    """

    def __init__(self, summary, message, suppress=False):
        """Initializes the exception with a summary and message.

        :param summary: a very brief description of the exception.
        :type summary: str
        :param message: an extended description of the exception.
        :type message: str
        :param suppress: specifies whether (False) or not (True) to call the parent constructor
            (default False).
        :type suppress: bool, optional
        """
        self._summary = summary
        self._message = message
        if not suppress:
            Exception.__init__(self, self._summary + " " + self._message)

    def get_message(self):
        """Get the message of the exception.

        :return: the message of the exception.
        :rtype: str
        """
        return self._message

    def get_summary(self):
        """Get the summary of the exception.

        :return: the summary of the exception.
        :rtype: str
        """
        return self._summary


class ObjectsFirstException(PLTException):
    """Exception for when the user attempts to load ranks before first loading objects.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load ranks."
        message = "You have not loaded an objects file. Please load the objects " \
                  "data before loading the ranks data."
        PLTException.__init__(self, summary, message, suppress)


class RanksFormatException(PLTException):
    """Exception for when the user attempts to load ranks with an invalid format.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load ranks."
        message = "The ranks file you tried to load did not contain the correct amount " \
                  "of columns. Make sure you selected the appropriate parameters for loading the file " \
                  "and that the file you are uploading is in the correct format."
        PLTException.__init__(self, summary, message, suppress)


class IDsException(PLTException):
    """Exception for when the user attempts to load ranks containing anything other than IDs referring to objects.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load ranks."
        message = "Entries in a ranks file should only contain numeric IDs referring to " \
                  "objects (rows in the objects file)."
        PLTException.__init__(self, summary, message, suppress)


class ObjectIDsFormatException(PLTException):
    """Exception for when the user attempts to load objects containing non-numeric IDs.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load objects."
        message = "The ID column in an objects file should consist of numeric values only."
        PLTException.__init__(self, summary, message, suppress)


class NonNumericFeatureException(PLTException):
    """Exception for when the user attempts to load data containing non-numeric values.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load objects."
        message = "Non-numeric feature values are currently not supported. " \
                  "Please make sure your data set consists of numeric values only."
        PLTException.__init__(self, summary, message, suppress)


class ParamIgnoredWarning(UserWarning):
    """Custom warning to inform user that one of the parameters passed to a method or function is being ignored in
    favour of another parameter passed to the method/function which overrides the former.

    Extends :class:`UserWarning`.
    """


class DataSetValueWarning(UserWarning):
    """Custom warning to inform users that the data set they are loading contains values which
    are not entirely numerical (i.e., cannot be converted to float or int).

    Extends :class:`UserWarning`.
    """


class NormalizationValueError(PLTException):
    """Exception for when the user attempts normalization on values that cannot be converted to int or float.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, f_id, norm_method, f_name=None, suppress=False):
        """Set the exception details.

        :param f_id: ID number of the feature causing the error.
        :type f_id: int
        :param norm_method: the attempted normalization method.
        :type norm_method: :class:`pyplt.util.enums.NormalizationType`
        :param f_name: name of the feature causing the error (default None).
        :type f_name: str, optional
        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Could not carry out normalization."
        # The value '" + str(value) + "' for
        if f_name is not None:
            message = "The feature '" + str(f_name) + "' could not be normalized using the " + str(norm_method.name) +\
                      " method as some values are not entirely " \
                      "numeric (i.e., they could not be converted to float or int)."
        else:
            message = "Feature " + str(f_id) + " could not be normalized using the " + str(norm_method.name) +\
                      " method as some values are not entirely numeric " \
                      "(i.e., they could not be converted to float or int)."
        PLTException.__init__(self, summary, message, suppress)


class NoFeaturesError(PLTException):
    """Exception for when the user attempts to run an experiment without any features to represent the objects.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot run experiment."
        message = "You have not included any features to represent your data. " \
                  "Please make sure you include at least one feature."
        PLTException.__init__(self, summary, message, suppress)


class NoRanksDerivedError(PLTException):
    """Exception for when no ranks could be derived from the given ratings-based dataset (single file format).

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "No ranks derived."
        message = "No pairwise preferences could be derived from your data. This is either because there are no " \
                  "clear pairwise preferences in your data or because none of the clear pairwise preferences in your " \
                  "data conform to your chosen values for the rank derivation parameters (i.e., the minimum " \
                  "distance margin and the memory)."
        PLTException.__init__(self, summary, message, suppress)


class InvalidParameterValueException(PLTException):
    """Exception for when the user attempts to use an invalid value for the given parameter of an algorithm/method.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, parameter, value=None, method=None, is_algorithm=False, additional_msg="", suppress=False):
        """Set the exception details.

        :param parameter: the parameter for which the error was given.
        :type parameter: str
        :param value: the value (assigned to `parameter`) causing the error (default None).
        :type value: object, optional
        :param is_algorithm: specifies whether the parameter belonged to an algorithm (True) or method (False)
            (default False).
        :type is_algorithm: bool, optional
        :param additional_msg: an additional message to include in the exception message.
        :type additional_msg: str, optional
        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot run experiment."
        if value is None:
            message = "You have entered an invalid value for the " + str(parameter) + " parameter"
            if method is not None:
                message += " of"
        else:
            message = "The " + str(parameter) + " value of " + str(value) + " is invalid"
            if method is not None:
                message += " for"
        if method is not None:
            if is_algorithm:
                message += (" the " + str(method) + " algorithm")
            else:
                message += (" the " + str(method) + " method")
        message += "."
        message += (" " + str(additional_msg))
        PLTException.__init__(self, summary, message, suppress)


class ManualFoldsFormatException(PLTException):
    """Exception for when the user attempts to load a manual folds file with an invalid format.

    A manual folds file is used to specify cross validation folds manually

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load folds."
        message = "The file you tried to load did not contain the correct amount " \
                  "of columns. Make sure you selected the appropriate parameters for loading the file " \
                  "and that the file you are uploading is in the correct format."
        PLTException.__init__(self, summary, message, suppress)


class FoldsSampleIDsException(PLTException):
    """Exception for when the user attempts to load a manual folds file with the wrong sample IDs.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load folds."
        message = "The sample IDs that you included in the manual folds file do not match your data."
        PLTException.__init__(self, summary, message, suppress)


class NonNumericValuesException(PLTException):
    """Exception for when the user attempts to load data containing non-numeric values.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load objects."
        message = "The file you loaded contains non-numeric data. " \
                  "Please make sure your data set consists of numeric values only."
        PLTException.__init__(self, summary, message, suppress)


class FoldsRowsException(PLTException):
    """Exception for when the user attempts to load a manual folds file with an invalid amount of rows.

    Applies for KFoldCrossValidation.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot load folds for cross validation."
        message = "The amount of rows you included in the manual folds file does not match the amount of samples in " \
                  "your data."
        PLTException.__init__(self, summary, message, suppress)


class IncompatibleFoldIndicesException(PLTException):
    """Exception for when the amount of user-specified fold indices does not match the amount of samples in the dataset.

    Applies for KFoldCrossValidation.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot split folds for cross validation."
        message = "The amount of indices you included in the manual folds parameter of K-Fold Cross " \
                  "Validation (KFCV) does not match the amount of samples in your data."
        PLTException.__init__(self, summary, message, suppress)


class MissingManualFoldsException(PLTException):
    """Exception for when the user chooses to specify folds manually without uploading a manual folds file.

    Applies for KFoldCrossValidation.

    Extends :class:`pyplt.exceptions.PLTException`.
    """

    def __init__(self, suppress=False):
        """Set the exception details.

        :param suppress: specifies whether (False) or not (True) to call the constructor of parent
            class :class:`Exception` (default False).
        :type suppress: bool
        """
        summary = "Cannot split folds for cross validation."
        message = "You chose to specify folds manually but failed to upload a file containing the fold IDs. " \
                  "To specify folds manually, use the 'Load Fold IDs' button in the KFCV menu to upload the " \
                  "necessary file."
        PLTException.__init__(self, summary, message, suppress)
