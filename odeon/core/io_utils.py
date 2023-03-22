"""Folder manager

This tool is responsible for the creation and destruction of folders, subfolders and files

Notes
-----

"""
import json
import os
import pathlib
import shutil
from os import listdir
from typing import Dict, List

import yaml

from .exceptions import ErrorCodes, OdeonError
from .types import URI


def create_folder(path: str,
                  parents: bool = True,
                  exist_ok: bool = True):
    """create folder with the whole hierarchy if required

    Parameters
    ----------
    path complete path of the folder

    Returns
    -------

    """
    pathlib.Path(path).mkdir(parents=parents,
                             exist_ok=exist_ok)


def find_file_names(path_to_dir: str, suffix: str = ".csv"):
    """

    Parameters
    ----------
    path_to_dir str
    suffix str

    Returns
    -------
    list[str]: list of files
    """

    file_names = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in file_names if filename.endswith(suffix)]


def build_directories(paths, append: bool = True, exist_ok: bool = True):
    """
    Make directory
    Parameters
    ----------
    paths str

    Returns
    -------

    """

    for k, path in paths.items():

        if os.path.isdir(path) and bool(append) is False:

            shutil.rmtree(path)

        os.makedirs(path, exist_ok=exist_ok)


def save_dict_as_json(d: Dict, output_file: str) -> None:
    """

    :param d:
    :param output_file:
    :return:
    """
    with open(output_file, 'w') as fp:
        json.dump(d, fp)


def create_path_if_not_exists(path: URI, exist_ok: bool = True, parents: bool = True) -> None:
    """
    Create directory if not exists
    Parameters
    ----------
    path: str | Path
    exist_ok: bool
    parents: bool

    Returns
    -------
     None
    """
    path = pathlib.Path(path)
    if path.exists() is False:
        pathlib.Path(path).mkdir(exist_ok=exist_ok, parents=parents)


def list_raster_files(path: str, extensions: List[str]) -> List:
    """
    Parameters
    ----------
     path: str, a directory with absolute URI

    Returns
    -------
     A list of files
    """
    return [fn for fn in os.listdir(path)
            if any(fn.endswith(ext) for ext in extensions)]


def load_yaml_file(path: URI) -> Dict:

    with open(path, "r") as stream:
        try:
            return dict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            error_message = f'yaml file {path} doesn"t exist'
            raise OdeonError(error_code=ErrorCodes.ERR_FILE_NOT_EXIST,
                             message=error_message,
                             stack_trace=exc)


def save_yaml_file(path: URI, data: Dict) -> None:
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def create_empty_file(path: URI):
    with open(path, 'w'):
        pass
